from flask import Flask, render_template, jsonify, request
import cv2
import base64
import numpy as np
from deepface import DeepFace
import io
from PIL import Image
import threading
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
    
    def calculate_confidence(self, frame, age, emotions, dominant_emotion):
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
            if elapsed_time > 15.0 and len(session["ages"]) < 8:
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
                
                # Konfidenzberechnung basierend auf verschiedenen Faktoren
                confidence = self.calculate_confidence(frame, age, emotions, dominant_emotion)

                # Altersbereich basierend auf Konfidenz
                uncertainty = max(2, int((100 - confidence) / 10))  # Min 2, max 10 Jahre Unsicherheit
                age_range = f"{max(0, int(age - uncertainty))}-{int(age + uncertainty)}"

                # Gültiges Alter hinzufügen
                session["ages"].append(age)
                session["frame_details"].append({
                    "frame_number": len(session["ages"]),
                    "age": float(round(age, 1)),  # Explizit zu Python float konvertieren
                    "age_range": age_range,
                    "confidence": float(round(confidence, 1)),  # Explizit zu Python float konvertieren
                    "emotion": str(dominant_emotion),  # Explizit zu Python string konvertieren
                    "emotion_confidence": float(round(emotions[dominant_emotion], 1)),  # Explizit zu Python float konvertieren
                    "timestamp": float(time.time() - session["start_time"])  # Explizit zu Python float konvertieren
                })

                return {
                    "status": "processing",
                    "frames_collected": len(session["ages"]),
                    "frames_needed": 8 - len(session["ages"]),
                    "attempted": session["attempted"],
                    "rejected": session["rejected"],
                    "elapsed": elapsed_time,
                    "last_age": int(round(age)),  # Explizit zu Python int konvertieren
                    "frame_details": session["frame_details"]
                }
                
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