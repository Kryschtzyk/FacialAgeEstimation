import cv2
from deepface import DeepFace
import numpy as np


def detect_face_occlusion(frame):
    """
    Erkennt mögliche Gesichtsverdeckungen durch verschiedene Methoden
    """
    try:
        # Analysiere das Gesicht mit DeepFace um zusätzliche Informationen zu bekommen
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Konvertiere zu Graustufen für weitere Analysen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Lade Haar-Cascade Klassifikatoren für verschiedene Objekte
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Erkenne Gesichter
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return True  # Kein klares Gesicht erkannt
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            
            # Erkenne Augen im Gesichtsbereich
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            # Wenn weniger als 2 Augen erkannt werden, könnte das Gesicht verdeckt sein
            if len(eyes) < 2:
                return True
            
            # Prüfe auf ungewöhnliche Helligkeitsverteilung (könnte auf Maske hindeuten)
            mean_brightness = np.mean(roi_gray)
            std_brightness = np.std(roi_gray)
            
            # Sehr niedrige Standardabweichung könnte auf gleichmäßige Verdeckung hindeuten
            if std_brightness < 20:
                return True
        
        return False
        
    except Exception as e:
        print(f"Fehler bei der Verdeckungserkennung: {e}")
        return True  # Im Zweifelsfall als verdeckt behandeln


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Prüfe auf Gesichtsverdeckungen
            is_occluded = detect_face_occlusion(frame)
            
            if is_occluded:
                # Zeige Meldung bei verdecktem Gesicht
                cv2.putText(
                    frame,
                    "Gesicht verdeckt - kein Alter verfügbar",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),  # Rot
                    2,
                    cv2.LINE_AA,
                )
            else:
                # Analysiere das Alter nur wenn das Gesicht nicht verdeckt ist
                try:
                    result = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)
                    age = result[0]['age']

                    cv2.putText(
                        frame,
                        f"Age: {int(age)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),  # Grün
                        2,
                        cv2.LINE_AA,
                    )
                except Exception as e:
                    cv2.putText(
                        frame,
                        "Alter nicht erkennbar",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),  # Orange
                        2,
                        cv2.LINE_AA,
                    )

            cv2.imshow("Age Estimation", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()