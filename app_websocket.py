from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
from deepface import DeepFace
import io
from PIL import Image
import time
import threading
import logging
from collections import deque
import json
import os

# Logging konfigurieren
log_dir = 'logs' if os.path.exists('logs') else '/app/logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'age_estimation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class WebSocketAgeEstimator:
    def __init__(self):
        self.session_data = {}
        self.camera_stream = None
        self.is_streaming = False
        self.camera_thread = None

    def detect_face_quick(self, frame):
        """Schnelle Gesichtserkennung ohne DeepFace"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) != 1:
                return False, "Genau ein Gesicht benötigt"

            x, y, w, h = faces[0]
            roi_gray = gray[y:y+h, x:x+w]

            # Gesichtsgröße prüfen
            frame_area = frame.shape[0] * frame.shape[1]
            face_area = w * h
            face_ratio = face_area / frame_area

            if face_ratio < 0.05:
                return False, "Näher zur Kamera"
            if face_ratio > 0.4:
                return False, "Weiter von der Kamera"

            # Augen prüfen
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) < 2:
                return False, "Beide Augen müssen sichtbar sein"

            # Helligkeit prüfen
            std_brightness = np.std(roi_gray)
            if std_brightness < 15:
                return False, "Gesicht möglicherweise verdeckt"

            return True, "Gesicht OK"

        except Exception as e:
            logger.error(f"Fehler bei Gesichtserkennung: {str(e)}")
            return False, f"Fehler: {str(e)[:30]}"

    def detect_spoofing_heuristic(self, frame):
        """Heuristische Spoofing-Erkennung"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Texture-Analyse
            texture_variance = np.var(cv2.Laplacian(gray, cv2.CV_64F))

            # Farbverteilung
            color_variance = np.var(frame, axis=(0,1))
            color_balance = np.std(color_variance)

            # Kantenschärfe
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            # Bewertung der Faktoren
            spoofing_score = 0.8

            if texture_variance < 100:
                spoofing_score -= 0.2
            elif texture_variance > 500:
                spoofing_score += 0.1

            if color_balance < 10:
                spoofing_score -= 0.15
            elif color_balance > 20:
                spoofing_score += 0.05

            if edge_density < 0.05:
                spoofing_score -= 0.1
            elif edge_density > 0.2:
                spoofing_score -= 0.15

            spoofing_score = max(0.3, min(0.95, spoofing_score))
            is_real = spoofing_score > 0.6

            logger.info(f"Spoofing-Analyse: Echtheit={is_real}, Score={spoofing_score:.3f}, "
                       f"Textur={texture_variance:.1f}, Farbe={color_balance:.1f}, Kanten={edge_density:.3f}")

            return is_real, spoofing_score

        except Exception as e:
            logger.error(f"Fehler bei Spoofing-Erkennung: {str(e)}")
            return True, 0.7

    def calculate_confidence(self, frame, age, emotions, dominant_emotion, spoofing_data=None):
        """Berechnet die Konfidenz der Altersschätzung"""
        confidence = 100.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Bildqualitätsfaktoren
        brightness = np.mean(gray)
        if brightness < 50 or brightness > 200:
            confidence -= 15

        contrast = np.std(gray)
        if contrast < 30:
            confidence -= 10

        # Emotionsstabilität
        neutral_score = emotions.get('neutral', 0)
        if neutral_score > 50:
            confidence += 10
        elif dominant_emotion in ['angry', 'fear', 'disgust', 'surprise']:
            confidence -= 15

        # Altersplausibilität
        if 10 <= age <= 70:
            confidence += 5
        elif age < 10 or age > 80:
            confidence -= 10

        # Bildschärfe
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            confidence -= 15
        elif laplacian_var > 500:
            confidence += 5

        # Anti-Spoofing Faktor
        if spoofing_data:
            is_real, spoofing_confidence = spoofing_data
            if not is_real:
                confidence -= 30
            elif spoofing_confidence > 0.8:
                confidence += 10
            elif spoofing_confidence < 0.5:
                confidence -= 20

        return max(40.0, min(99.0, confidence))

    def process_age_estimation(self, frame, session_id):
        """Verarbeitet die Altersschätzung für einen Frame"""
        try:
            # Session initialisieren
            if session_id not in self.session_data:
                self.session_data[session_id] = {
                    "ages": [],
                    "frame_details": [],
                    "start_time": time.time(),
                    "attempted": 0,
                    "rejected": 0
                }

            session = self.session_data[session_id]
            session["attempted"] += 1

            # Schnelle Validierung
            is_valid, reason = self.detect_face_quick(frame)
            if not is_valid:
                session["rejected"] += 1
                logger.warning(f"Frame abgelehnt: {reason}")
                return {
                    "status": "invalid",
                    "customer_message": reason,
                    "reason": reason
                }

            # Anti-Spoofing Check
            is_real, spoofing_confidence = self.detect_spoofing_heuristic(frame)
            if not is_real:
                session["rejected"] += 1
                logger.warning(f"Spoofing erkannt: Konfidenz={spoofing_confidence:.3f}")
                return {
                    "status": "invalid",
                    "customer_message": "Bitte verwenden Sie ein echtes Gesicht",
                    "reason": f"Spoofing-Verdacht: {spoofing_confidence*100:.1f}%"
                }

            # DeepFace Analyse
            emotion_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotions = emotion_result[0]['emotion']
            dominant_emotion = emotion_result[0]['dominant_emotion']

            # Emotionsprüfung
            extreme_emotions = ['angry', 'disgust', 'fear', 'surprise']
            if dominant_emotion in extreme_emotions:
                emotion_value = emotions[dominant_emotion]
                if emotion_value > 60:
                    session["rejected"] += 1
                    logger.warning(f"Extreme Emotion erkannt: {dominant_emotion} ({emotion_value:.1f}%)")
                    return {
                        "status": "invalid",
                        "customer_message": "Bitte schauen Sie neutral in die Kamera",
                        "reason": f"Extreme Emotion: {dominant_emotion}"
                    }

            # Altersanalyse
            age_result = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)
            age = float(age_result[0]['age'])

            # Plausibilitätscheck
            if age < 5 or age > 100:
                session["rejected"] += 1
                logger.warning(f"Unplausibles Alter: {age:.1f}")
                return {
                    "status": "invalid",
                    "customer_message": "Bitte positionieren Sie sich korrekt vor der Kamera",
                    "reason": f"Unplausibles Alter: {age:.1f}"
                }

            # Konfidenzberechnung
            confidence = self.calculate_confidence(frame, age, emotions, dominant_emotion, (is_real, spoofing_confidence))

            # Frame Details für Logging
            frame_detail = {
                "frame_number": len(session["ages"]) + 1,
                "age": age,
                "confidence": confidence,
                "emotion": dominant_emotion,
                "emotion_confidence": emotions[dominant_emotion],
                "is_real": is_real,
                "spoofing_confidence": spoofing_confidence,
                "timestamp": time.time() - session["start_time"]
            }

            # Detailliertes Logging
            logger.info(f"Frame #{frame_detail['frame_number']} - "
                       f"Alter: {age:.1f}, Konfidenz: {confidence:.1f}%, "
                       f"Emotion: {dominant_emotion} ({emotions[dominant_emotion]:.1f}%), "
                       f"Echtheit: {is_real} ({spoofing_confidence:.3f})")

            session["ages"].append(age)
            session["frame_details"].append(frame_detail)

            # Prüfe ob genug Frames gesammelt
            if len(session["ages"]) >= 8:
                # Finales Ergebnis berechnen
                ages = [detail["age"] for detail in session["frame_details"]]
                if len(ages) >= 5:
                    ages_sorted = sorted(ages)
                    ages_sorted = ages_sorted[1:-1]  # Ausreißer entfernen
                    final_age = int(np.mean(ages_sorted))
                else:
                    final_age = int(np.mean(ages))

                avg_confidence = np.mean([detail["confidence"] for detail in session["frame_details"]])
                avg_spoofing = np.mean([detail["spoofing_confidence"] for detail in session["frame_details"]])

                # Finales Logging
                logger.info(f"FINALE ANALYSE - Alter: {final_age}, "
                           f"Durchschnittskonfidenz: {avg_confidence:.1f}%, "
                           f"Durchschnittliche Echtheit: {avg_spoofing:.3f}, "
                           f"Frames verwendet: {len(session['ages'])}/{session['attempted']}")

                return {
                    "status": "complete",
                    "age": final_age,
                    "customer_message": f"Altersschätzung abgeschlossen",
                    "frame_count": len(session["ages"]),
                    "attempted": session["attempted"],
                    "rejected": session["rejected"]
                }

            return {
                "status": "processing",
                "customer_message": f"Analyse läuft... ({len(session['ages'])}/8 Frames)",
                "frames_collected": len(session["ages"]),
                "frames_needed": 8 - len(session["ages"])
            }

        except Exception as e:
            logger.error(f"Fehler bei Altersschätzung: {str(e)}")
            return {
                "status": "error",
                "customer_message": "Technischer Fehler - Bitte erneut versuchen",
                "reason": str(e)
            }

    def start_camera_stream(self):
        """Startet den Kamera-Stream mit verbesserter Mac-to-Docker Streaming-Unterstützung"""
        try:
            # Netzwerk-Debugging
            logger.info("Netzwerk-Debug-Informationen:")
            try:
                import socket
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                logger.info(f"Hostname: {hostname}, Container-IP: {local_ip}")
            except Exception as e:
                logger.warning(f"Konnte Netzwerkinformationen nicht abrufen: {e}")

            # Mac Camera Server Adresse aus Umgebungsvariablen oder Standardwerten lesen
            mac_host = os.environ.get("MAC_CAMERA_HOST", "192.168.178.155")
            mac_port = os.environ.get("MAC_CAMERA_PORT", "5002")

            # NUR MJPEG-Stream Optionen versuchen, keine lokalen Kameras mehr
            camera_options = [
                f"http://{mac_host}:{mac_port}/stream",    # Direkte Mac-IP aus Umgebungsvariable
                "http://host.docker.internal:5002/stream", # Docker Desktop Host
                "http://mac-host:5002/stream",             # Host-Eintrag aus docker-compose
                "http://172.17.0.1:5002/stream",           # Docker Default Gateway
                "http://192.168.178.155:5002/stream",      # Explizite Mac-IP
                "http://10.147.59.72:5002/stream",         # Alternative Mac-IP
                "http://localhost:5002/stream",            # Lokaler Test (wenn Docker im Host-Netzwerk)
                "http://127.0.0.1:5002/stream",            # Lokaler Test Alternative
            ]

            logger.info(f"Suche nach Kamera-Stream, primäre URL: {camera_options[0]}")

            for camera in camera_options:
                logger.info(f"Versuche MJPEG-Stream von {camera}...")
                try:
                    # MJPEG-Stream über Netzwerk
                    self.camera_stream = cv2.VideoCapture(camera)

                    # Kurze Wartezeit hinzufügen für Netzwerkverbindungsaufbau
                    time.sleep(0.5)

                    if self.camera_stream.isOpened():
                        # Test ob Frame gelesen werden kann
                        ret, test_frame = self.camera_stream.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            logger.info(f"✅ Kamera {camera} erfolgreich geöffnet")
                            break
                        else:
                            logger.warning(f"Kamera {camera} geöffnet, aber kein gültiges Frame erhalten")
                            self.camera_stream.release()
                            self.camera_stream = None
                    else:
                        logger.warning(f"Kamera {camera} konnte nicht geöffnet werden")

                except Exception as e:
                    logger.warning(f"Fehler beim Öffnen der Kamera {camera}: {str(e)}")
                    if self.camera_stream:
                        self.camera_stream.release()
                        self.camera_stream = None

            # Wenn keine Netzwerkkamera gefunden wurde, versuche nicht mehr die lokalen Kameras

            if not self.camera_stream or not self.camera_stream.isOpened():
                logger.error("Keine funktionierende Kamera gefunden")
                logger.error("Bitte überprüfe, ob der Mac Camera Server läuft und über das Netzwerk erreichbar ist")
                logger.error(f"Mac Camera Server sollte unter http://{mac_host}:{mac_port}/stream laufen")
                emit('camera_status', {
                    'status': 'error',
                    'message': 'Keine Kamera erkannt. Mac Camera Server läuft nicht oder ist nicht erreichbar.'
                })
                return False

            # Teste nochmals das Frame-Lesen
            ret, test_frame = self.camera_stream.read()
            if not ret or test_frame is None:
                logger.error("Kamera geöffnet, aber Frame-Test fehlgeschlagen")
                self.camera_stream.release()
                return False

            logger.info(f"Kamera-Auflösung: {test_frame.shape[1]}x{test_frame.shape[0]}")

            self.is_streaming = True
            logger.info("Kamera-Stream erfolgreich gestartet")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Starten der Kamera: {str(e)}")
            if self.camera_stream:
                self.camera_stream.release()
                self.camera_stream = None
            return False

    def stop_camera_stream(self):
        """Stoppt den Kamera-Stream"""
        self.is_streaming = False
        if self.camera_stream:
            self.camera_stream.release()
            self.camera_stream = None
        logger.info("Kamera-Stream gestoppt")

    def get_frame(self):
        """Holt einen Frame von der Kamera"""
        if not self.camera_stream or not self.is_streaming:
            return None

        ret, frame = self.camera_stream.read()
        if ret:
            return frame
        return None

estimator = WebSocketAgeEstimator()

@app.route('/')
def index():
    return render_template('kiosk.html')

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client verbunden: {request.sid}")
    emit('status', {'message': 'Verbindung hergestellt'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client getrennt: {request.sid}")

@socketio.on('start_camera')
def handle_start_camera():
    """Startet die Kamera und den Stream"""
    if estimator.start_camera_stream():
        emit('camera_status', {'status': 'started', 'message': 'Kamera gestartet'})
        # Starte Camera Thread
        if not estimator.camera_thread or not estimator.camera_thread.is_alive():
            estimator.camera_thread = threading.Thread(target=camera_stream_thread, daemon=True)
            estimator.camera_thread.start()
    else:
        emit('camera_status', {'status': 'error', 'message': 'Kamera konnte nicht gestartet werden'})

@socketio.on('stop_camera')
def handle_stop_camera():
    """Stoppt die Kamera"""
    estimator.stop_camera_stream()
    emit('camera_status', {'status': 'stopped', 'message': 'Kamera gestoppt'})

@socketio.on('start_analysis')
def handle_start_analysis(data):
    """Startet die Altersanalyse"""
    session_id = data.get('session_id', 'default')
    logger.info(f"Analyse gestartet für Session: {session_id}")
    emit('analysis_status', {
        'status': 'started',
        'message': 'Bitte schauen Sie direkt in die Kamera',
        'session_id': session_id
    })

@socketio.on('process_frame')
def handle_process_frame(data):
    """Verarbeitet einen einzelnen Frame"""
    try:
        session_id = data.get('session_id', 'default')
        image_data = data.get('image')

        if not image_data:
            emit('analysis_result', {
                'status': 'error',
                'customer_message': 'Kein Bild empfangen'
            })
            return

        # Base64 zu Frame konvertieren
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Altersschätzung durchführen
        result = estimator.process_age_estimation(frame, session_id)

        # Ergebnis an Client senden
        emit('analysis_result', result)

    except Exception as e:
        logger.error(f"Fehler bei Frame-Verarbeitung: {str(e)}")
        emit('analysis_result', {
            'status': 'error',
            'customer_message': 'Technischer Fehler'
        })

@socketio.on('reset_session')
def handle_reset_session(data):
    """Setzt eine Session zurück"""
    session_id = data.get('session_id', 'default')
    if session_id in estimator.session_data:
        del estimator.session_data[session_id]
    logger.info(f"Session zurückgesetzt: {session_id}")
    emit('session_reset', {'session_id': session_id})

def camera_stream_thread():
    """Thread für kontinuierlichen Kamera-Stream"""
    while estimator.is_streaming:
        frame = estimator.get_frame()
        if frame is not None:
            # Frame zu Base64 konvertieren
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_data = base64.b64encode(buffer).decode('utf-8')

            # Frame an alle verbundenen Clients senden
            socketio.emit('camera_frame', {
                'image': f'data:image/jpeg;base64,{frame_data}',
                'timestamp': time.time()
            })

        time.sleep(1/15)  # 15 FPS

if __name__ == '__main__':
    logger.info("WebSocket Altersschätzung Server gestartet")
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)
