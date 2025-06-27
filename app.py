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
    
    def process_frame(self, image_data, session_id):
        """Verarbeitet ein einzelnes Bild aus der Base64-Codierung"""
        try:
            # Prüfe, ob die Sitzung existiert und initialisiere ggf.
            if session_id not in self.session_data:
                self.session_data[session_id] = {
                    "ages": [],
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
                    "elapsed": elapsed_time
                }
            
            # Prüfe ob 8 gültige Frames erreicht wurden
            if len(session["ages"]) >= 8:
                # Berechne Durchschnitt und entferne Ausreißer
                ages = session["ages"]
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
                    "elapsed": elapsed_time
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
                
                # Age analysis
                age_result = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)
                age = age_result[0]['age']
                
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
                
                # Gültiges Alter hinzufügen
                session["ages"].append(age)
                
                return {
                    "status": "processing",
                    "frames_collected": len(session["ages"]),
                    "frames_needed": 8 - len(session["ages"]),
                    "attempted": session["attempted"],
                    "rejected": session["rejected"],
                    "elapsed": elapsed_time,
                    "last_age": round(age)
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