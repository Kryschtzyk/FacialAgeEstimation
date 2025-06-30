from flask import Flask, render_template, jsonify, request
import cv2
import base64
import numpy as np
from deepface import DeepFace
import io
from PIL import Image
import time
from collections import deque

app = Flask(__name__)

class WebAgeEstimator:
    def __init__(self):
        self.valid_ages = deque(maxlen=8)
        self.session_data = {}  # Speichert Sitzungsdaten

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
            return False, f"Fehler: {str(e)[:30]}"

    def detect_spoofing(self, frame):
        """Erkennt Spoofing-Versuche mit DeepFace"""
        try:
            # DeepFace Anti-Spoofing Analyse
            # Hinweis: Dies funktioniert mit neueren DeepFace Versionen
            spoofing_result = DeepFace.extract_faces(
                frame,
                anti_spoofing=True,
                enforce_detection=False
            )

            if len(spoofing_result) > 0:
                # Der Anti-Spoofing Score ist im Ergebnis enthalten
                face_data = spoofing_result[0]
                if hasattr(face_data, 'is_real') or 'is_real' in face_data:
                    is_real = face_data.get('is_real', True) if isinstance(face_data, dict) else getattr(face_data, 'is_real', True)
                    confidence = face_data.get('antispoof_score', 0.8) if isinstance(face_data, dict) else getattr(face_data, 'antispoof_score', 0.8)
                    return is_real, confidence

            # Fallback: Einfache heuristische Spoofing-Erkennung
            return self.detect_spoofing_heuristic(frame)

        except Exception as e:
            # Fallback bei Fehlern
            return self.detect_spoofing_heuristic(frame)

    def detect_spoofing_heuristic(self, frame):
        """Heuristische Spoofing-Erkennung als Fallback"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Texture-Analyse (echte Gesichter haben mehr Textur-Variation)
            texture_variance = np.var(cv2.Laplacian(gray, cv2.CV_64F))

            # 2. Farbverteilung (echte Gesichter haben natürlichere Farbverteilung)
            color_variance = np.var(frame, axis=(0,1))
            color_balance = np.std(color_variance)

            # 3. Kantenschärfe (Fotos haben oft gleichmäßigere Kanten)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            # Bewertung der Faktoren
            spoofing_score = 0.8  # Startwert (80% echt)

            if texture_variance < 100:
                spoofing_score -= 0.2  # Wenig Textur -> möglicherweise Foto
            elif texture_variance > 500:
                spoofing_score += 0.1  # Viel Textur -> wahrscheinlich echt

            if color_balance < 10:
                spoofing_score -= 0.15  # Unnatürliche Farbverteilung
            elif color_balance > 20:
                spoofing_score += 0.05  # Natürliche Farbverteilung

            if edge_density < 0.05:
                spoofing_score -= 0.1  # Sehr wenige Kanten
            elif edge_density > 0.2:
                spoofing_score -= 0.15  # Zu viele Kanten (möglicherweise Bildschirm)

            # Begrenzen auf 0.3 bis 0.95
            spoofing_score = max(0.3, min(0.95, spoofing_score))
            is_real = spoofing_score > 0.6

            return is_real, spoofing_score

        except Exception as e:
            # Bei Fehlern konservativ annehmen, dass es echt ist
            return True, 0.7

    def calculate_confidence(self, frame, age, emotions, dominant_emotion, spoofing_data=None):
        """Berechnet die Konfidenz der Altersschätzung basierend auf verschiedenen Faktoren"""
        confidence = 100.0  # Startwert

        # Bildqualitätsfaktoren
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Helligkeit und Kontrast
        brightness = np.mean(gray)
        if brightness < 50 or brightness > 200:
            confidence -= 15  # Schlechte Beleuchtung

        contrast = np.std(gray)
        if contrast < 30:
            confidence -= 10  # Geringer Kontrast

        # 2. Gesichtsgröße (bereits in detect_face_quick geprüft, aber für Konfidenz relevant)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_area = w * h
            frame_area = frame.shape[0] * frame.shape[1]
            face_ratio = face_area / frame_area

            if face_ratio < 0.1:
                confidence -= 20  # Gesicht zu klein
            elif face_ratio > 0.3:
                confidence -= 10  # Gesicht sehr groß
            elif 0.15 <= face_ratio <= 0.25:
                confidence += 5   # Optimale Gesichtsgröße

        # 3. Emotionsstabilität
        neutral_score = emotions.get('neutral', 0)
        if neutral_score > 50:
            confidence += 10  # Neutrale Mimik ist gut
        elif dominant_emotion in ['angry', 'fear', 'disgust', 'surprise']:
            confidence -= 15  # Extreme Emotionen reduzieren Konfidenz

        # 4. Altersplausibilität (interne Konsistenz)
        # DeepFace hat interne Unsicherheiten, simuliere diese
        if 10 <= age <= 70:
            confidence += 5   # Typischer Altersbereich
        elif age < 10 or age > 80:
            confidence -= 10  # Extremere Altersbereiche sind unsicherer

        # 5. Bildschärfe (grobe Approximation über Gradientstärke)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            confidence -= 15  # Unscharfes Bild
        elif laplacian_var > 500:
            confidence += 5   # Scharfes Bild

        # 6. Anti-Spoofing Faktor (NEU)
        if spoofing_data:
            is_real, spoofing_confidence = spoofing_data
            if not is_real:
                confidence -= 30  # Starke Reduktion bei Spoofing-Verdacht
            elif spoofing_confidence > 0.8:
                confidence += 10  # Bonus für hohe Echtheit
            elif spoofing_confidence < 0.5:
                confidence -= 20  # Reduktion bei niedriger Echtheit

        # Konfidenz zwischen 40 und 99 begrenzen
        return max(40.0, min(99.0, confidence))

    def process_frame(self, image_data, session_id):
        """Verarbeitet ein einzelnes Bild aus der Base64-Codierung"""
        try:
            # Prüfe, ob die Sitzung existiert und initialisiere ggf.
            if session_id not in self.session_data:
                self.session_data[session_id] = {
                    "ages": [],
                    "frame_details": [],  # Detaillierte Informationen pro Frame
                    "start_time": time.time(),
                    "attempted": 0,
                    "rejected": 0
                }

            session = self.session_data[session_id]
            session["attempted"] += 1

            # Prüfe Timeout
            elapsed_time = time.time() - session["start_time"]
            if elapsed_time > 60.0 and len(session["ages"]) < 8:
                return {
                    "status": "failed",
                    "reason": f"Timeout nach 15s - Nur {len(session['ages'])}/8 Frames gesammelt",
                    "attempted": session["attempted"],
                    "rejected": session["rejected"],
                    "elapsed": elapsed_time,
                    "frame_details": session["frame_details"]
                }

            # Prüfe ob 8 gültige Frames erreicht wurden
            if len(session["ages"]) >= 8:
                # Berechne Durchschnitt und entferne Ausreißer
                ages = [detail["age"] for detail in session["frame_details"]]
                if len(ages) >= 5:
                    ages_sorted = sorted(ages)
                    ages_sorted = ages_sorted[1:-1]  # Entferne niedrigste und höchste
                    final_age = int(np.mean(ages_sorted))
                else:
                    final_age = int(np.mean(ages))

                return {
                    "status": "complete",
                    "age": final_age,
                    "frame_count": len(session["ages"]),
                    "attempted": session["attempted"],
                    "rejected": session["rejected"],
                    "elapsed": elapsed_time,
                    "frame_details": session["frame_details"]
                }

            # Base64 zu Bild konvertieren
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Schnelle Validierung
            is_valid, reason = self.detect_face_quick(frame)
            if not is_valid:
                session["rejected"] += 1
                return {
                    "status": "invalid",
                    "reason": reason,
                    "attempted": session["attempted"],
                    "rejected": session["rejected"],
                    "elapsed": elapsed_time,
                    "frames_collected": len(session["ages"])
                }

            # DeepFace Analyse
            try:
                # Anti-Spoofing Check (NEU)
                is_real, spoofing_confidence = self.detect_spoofing(frame)
                if not is_real:
                    session["rejected"] += 1
                    return {
                        "status": "invalid",
                        "reason": f"Spoofing-Verdacht: Bitte verwenden Sie ein echtes Gesicht (Spoofing: {spoofing_confidence*100:.1f}%)",
                        "attempted": session["attempted"],
                        "rejected": session["rejected"],
                        "elapsed": elapsed_time,
                        "frames_collected": len(session["ages"])
                    }

                # Warnung bei niedriger Spoofing-Konfidenz (aber noch akzeptabel)
                if spoofing_confidence < 0.7:
                    # Frame wird akzeptiert, aber mit Warnung
                    spoofing_warning = f"Niedrige Echtheit ({spoofing_confidence*100:.1f}%)"
                else:
                    spoofing_warning = None

                # Emotion check für Grimassen
                emotion_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotions = emotion_result[0]['emotion']
                dominant_emotion = emotion_result[0]['dominant_emotion']

                # Prüfe auf extreme Emotionen (Grimassen)
                extreme_emotions = ['angry', 'disgust', 'fear', 'surprise']
                if dominant_emotion in extreme_emotions:
                    emotion_value = emotions[dominant_emotion]
                    if emotion_value > 60:
                        session["rejected"] += 1
                        return {
                            "status": "invalid",
                            "reason": f"Neutrale Mimik erforderlich ({dominant_emotion} {emotion_value:.0f}%)",
                            "attempted": session["attempted"],
                            "rejected": session["rejected"],
                            "elapsed": elapsed_time,
                            "frames_collected": len(session["ages"])
                        }

                # Prüfe auch auf zu starke Freude (übertriebenes Lachen)
                if dominant_emotion == 'happy' and emotions['happy'] > 85:
                    session["rejected"] += 1
                    return {
                        "status": "invalid",
                        "reason": f"Bitte lächeln Sie weniger stark ({emotions['happy']:.0f}%)",
                        "attempted": session["attempted"],
                        "rejected": session["rejected"],
                        "elapsed": elapsed_time,
                        "frames_collected": len(session["ages"])
                    }

                # Age analysis mit Konfidenzberechnung
                age_result = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)
                age = float(age_result[0]['age'])  # Konvertiere zu Python float

                # Validiere Alter (Plausibilitätscheck)
                if age < 5 or age > 100:
                    session["rejected"] += 1
                    return {
                        "status": "invalid",
                        "reason": f"Unplausibles Alter erkannt: {age:.1f}",
                        "attempted": session["attempted"],
                        "rejected": session["rejected"],
                        "elapsed": elapsed_time,
                        "frames_collected": len(session["ages"])
                    }

                # Konfidenzberechnung basierend auf verschiedenen Faktoren (inklusive Anti-Spoofing)
                confidence = self.calculate_confidence(frame, age, emotions, dominant_emotion, (is_real, spoofing_confidence))

                # Altersbereich basierend auf Konfidenz
                uncertainty = max(2, int((100 - confidence) / 10))  # Min 2, max 10 Jahre Unsicherheit
                age_range = f"{max(0, int(age - uncertainty))}-{int(age + uncertainty)}"

                # Gültiges Alter hinzufügen
                session["ages"].append(age)
                session["frame_details"].append({
                    "frame_number": len(session["ages"]),
                    "age": float(round(age, 1)),  # Explizit zu Python float konvertieren
                    "age_range": age_range,
                    "emotion": str(dominant_emotion),  # Explizit zu Python string konvertieren
                    "emotion_confidence": float(round(emotions[dominant_emotion], 1)),  # Explizit zu Python float konvertieren
                    "timestamp": float(time.time() - session["start_time"]),  # Explizit zu Python float konvertieren
                    "is_real": bool(is_real),  # Anti-Spoofing Ergebnis
                    "spoofing_confidence": float(round(spoofing_confidence, 3)),  # Anti-Spoofing Konfidenz
                    "spoofing_warning": spoofing_warning  # Warnung bei niedriger Echtheit
                })

                response_data = {
                    "status": "processing",
                    "frames_collected": len(session["ages"]),
                    "frames_needed": 8 - len(session["ages"]),
                    "attempted": session["attempted"],
                    "rejected": session["rejected"],
                    "elapsed": elapsed_time,
                    "last_age": int(round(age)),  # Explizit zu Python int konvertieren
                    "frame_details": session["frame_details"],
                    # Live-Spoofing-Daten für kontinuierliche Anzeige
                    "live_spoofing": {
                        "is_real": bool(is_real),
                        "confidence": float(round(spoofing_confidence, 3)),
                        "confidence_percent": float(round(spoofing_confidence * 100, 1))
                    }
                }

                # Füge Spoofing-Warnung hinzu, falls vorhanden
                if spoofing_warning:
                    response_data["spoofing_warning"] = spoofing_warning

                return response_data

            except Exception as e:
                session["rejected"] += 1
                return {
                    "status": "error",
                    "reason": f"DeepFace Fehler: {str(e)[:50]}",
                    "attempted": session["attempted"],
                    "rejected": session["rejected"],
                    "elapsed": elapsed_time,
                    "frames_collected": len(session["ages"])
                }

        except Exception as e:
            return {"status": "error", "reason": str(e)[:100]}

estimator = WebAgeEstimator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Verarbeitet einen Frame vom Client"""
    data = request.json
    image_data = data.get('image')
    session_id = data.get('session_id', 'default')

    if not image_data:
        return jsonify({"status": "error", "reason": "Kein Bild übertragen"})

    result = estimator.process_frame(image_data, session_id)
    return jsonify(result)

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Setzt eine Sitzung zurück"""
    data = request.json
    session_id = data.get('session_id', 'default')
    if session_id in estimator.session_data:
        del estimator.session_data[session_id]
    return jsonify({"status": "reset"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
