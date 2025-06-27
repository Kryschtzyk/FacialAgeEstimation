import cv2
from deepface import DeepFace
import numpy as np
import threading
import time
from collections import deque
import queue


class AgeEstimator:
    def __init__(self):
        self.valid_ages = deque(maxlen=8)
        self.processing = False
        self.final_age = None
        self.start_time = None
        self.max_duration = 10.0  # 10 Sekunden
        self.frame_queue = queue.Queue(maxsize=20)
        self.processing_thread = None
        self.stop_processing = False
        self.last_frame_time = 0
        self.frame_interval = 0.3  # Alle 300ms einen Frame verarbeiten
        self.current_status = "waiting"  # waiting, ready, processing, complete
        self.status_message = ""
        
    def detect_face_quick(self, frame):
        """
        Schnelle Gesichtserkennung ohne DeepFace für UI-Responsivität
        """
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
            
            return True, "Bereit für Analyse"
            
        except Exception as e:
            return False, f"Fehler: {str(e)[:30]}"
    
    def process_frames_worker(self):
        """
        Worker-Thread für die zeitaufwändige DeepFace-Analyse
        """
        consecutive_failures = 0
        max_failures = 5
        
        while not self.stop_processing and self.processing:
            try:
                # Warte auf neuen Frame (mit Timeout)
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Prüfe ob wir bereits genug Frames haben
                if len(self.valid_ages) >= 8:
                    continue
                
                # Erst schnelle Validierung
                is_valid, _ = self.detect_face_quick(frame)
                if not is_valid:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        time.sleep(0.2)
                    continue
                
                # Dann aufwändige DeepFace-Analyse
                try:
                    # Emotion check für Grimassen
                    emotion_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    emotions = emotion_result[0]['emotion']
                    dominant_emotion = emotion_result[0]['dominant_emotion']
                    
                    # Prüfe auf extreme Emotionen
                    if dominant_emotion not in ['neutral', 'happy']:
                        emotion_value = emotions[dominant_emotion]
                        if emotion_value > 65:
                            consecutive_failures += 1
                            continue
                    
                    # Age analysis
                    age_result = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)
                    age = age_result[0]['age']
                    
                    # Gültiges Alter hinzufügen
                    self.valid_ages.append(age)
                    consecutive_failures = 0
                    
                    print(f"Frame {len(self.valid_ages)}/8 verarbeitet: Alter {age:.1f}")
                    
                except Exception as e:
                    print(f"DeepFace Fehler: {e}")
                    consecutive_failures += 1
                    continue
                
                # Pause zwischen Verarbeitungen
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Worker Thread Fehler: {e}")
                consecutive_failures += 1
                time.sleep(0.3)
    
    def start_processing(self):
        """
        Startet die Verarbeitung
        """
        if not self.processing:
            self.valid_ages.clear()
            self.processing = True
            self.start_time = time.time()
            self.final_age = None
            self.stop_processing = False
            self.current_status = "processing"
            
            # Starte Worker Thread
            self.processing_thread = threading.Thread(target=self.process_frames_worker)
            self.processing_thread.daemon = True
            self.processing_thread.start()
    
    def add_frame_for_processing(self, frame):
        """
        Fügt einen Frame zur Verarbeitung hinzu (non-blocking)
        """
        if not self.processing:
            return
            
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_frame_time >= self.frame_interval:
            try:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy(), block=False)
                else:
                    # Queue leeren und aktuellen Frame hinzufügen
                    try:
                        while True:
                            self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put(frame.copy(), block=False)
                
                self.last_frame_time = current_time
            except queue.Full:
                pass
    
    def update_status(self, frame):
        """
        Aktualisiert den Status basierend auf dem aktuellen Frame
        """
        current_time = time.time()
        
        if self.processing:
            # Prüfe ob Verarbeitung abgeschlossen ist
            frames_complete = len(self.valid_ages) >= 8
            time_complete = self.start_time and (current_time - self.start_time) >= self.max_duration
            
            if frames_complete or time_complete:
                self.current_status = "complete"
                self.processing = False
                self.stop_processing = True
            else:
                self.current_status = "processing"
        
        elif self.current_status == "complete":
            # Status bleibt auf complete bis Reset
            pass
        
        else:
            # Prüfe Gesichtsstatus (aber nur alle 500ms um Flackern zu vermeiden)
            if not hasattr(self, 'last_status_update'):
                self.last_status_update = 0
            
            if current_time - self.last_status_update > 0.5:
                is_valid, reason = self.detect_face_quick(frame)
                if is_valid:
                    self.current_status = "ready"
                    self.status_message = "Bereit für Analyse"
                else:
                    self.current_status = "waiting"
                    self.status_message = reason
                
                self.last_status_update = current_time
    
    def get_progress(self):
        """
        Gibt den aktuellen Fortschritt zurück (0.0 bis 1.0)
        """
        if not self.processing and self.current_status != "complete":
            return 0.0
        
        if self.current_status == "complete":
            return 1.0
        
        # Nur basierend auf gesammelten Frames
        return len(self.valid_ages) / 8.0
    
    def get_final_age(self):
        """
        Berechnet das finale Alter als Durchschnitt
        """
        if len(self.valid_ages) > 0 and self.final_age is None:
            self.final_age = int(np.mean(list(self.valid_ages)))
        return self.final_age
    
    def get_frame_count(self):
        """
        Gibt die Anzahl der gesammelten gültigen Frames zurück
        """
        return len(self.valid_ages)
    
    def get_elapsed_time(self):
        """
        Gibt die verstrichene Zeit zurück
        """
        if self.start_time:
            return time.time() - self.start_time
        return 0
    
    def reset(self):
        """
        Setzt den Estimator zurück
        """
        self.stop_processing = True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        # Queue leeren
        try:
            while True:
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        
        self.valid_ages.clear()
        self.processing = False
        self.final_age = None
        self.start_time = None
        self.processing_thread = None
        self.stop_processing = False
        self.current_status = "waiting"
        self.status_message = ""


def draw_progress_circle(frame, center, radius, progress, color=(0, 255, 0)):
    """
    Zeichnet einen Fortschrittskreis
    """
    # Äußerer Kreis (Hintergrund)
    cv2.circle(frame, center, radius, (100, 100, 100), 3)
    
    # Fortschrittskreis
    if progress > 0:
        angle = int(360 * progress)
        axes = (radius, radius)
        cv2.ellipse(frame, center, axes, -90, 0, angle, color, 3)
    
    # Innerer Text
    if progress >= 1.0:
        text = "OK"
        color_text = color
    else:
        text = f"{int(progress * 8)}/8"
        color_text = color
    
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + text_size[1] // 2
    
    cv2.putText(frame, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 2)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    estimator = AgeEstimator()
    show_result_until = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            
            # Aktualisiere Status
            estimator.update_status(frame)
            
            # Zeige Ergebnis für 5 Sekunden nach Abschluss
            if estimator.current_status == "complete" and show_result_until == 0:
                show_result_until = current_time + 5.0
            
            # Reset nach Anzeige des Ergebnisses
            if show_result_until > 0 and current_time > show_result_until:
                estimator.reset()
                show_result_until = 0
            
            # UI basierend auf Status
            if estimator.current_status == "complete" or show_result_until > 0:
                # Zeige finales Ergebnis
                final_age = estimator.get_final_age()
                frame_count = estimator.get_frame_count()
                cv2.putText(frame, f"Geschätztes Alter: {final_age} Jahre", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Basierend auf {frame_count} Frames", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                draw_progress_circle(frame, (frame.shape[1] - 80, 80), 50, 1.0, (0, 255, 0))
            
            elif estimator.current_status == "processing":
                # Zeige Verarbeitungsfortschritt
                progress = estimator.get_progress()
                frame_count = estimator.get_frame_count()
                elapsed = estimator.get_elapsed_time()
                
                draw_progress_circle(frame, (frame.shape[1] - 80, 80), 50, progress, (0, 255, 255))
                
                cv2.putText(frame, f"Sammle Frames... ({frame_count}/8)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Zeit: {elapsed:.1f}s / max 10.0s", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Füge Frame zur Verarbeitung hinzu
                estimator.add_frame_for_processing(frame)
            
            elif estimator.current_status == "ready":
                cv2.putText(frame, "Bereit für Analyse!", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "Drücken Sie SPACE zum Starten", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                draw_progress_circle(frame, (frame.shape[1] - 80, 80), 50, 0, (0, 255, 0))
            
            else:  # waiting
                cv2.putText(frame, f"Warten: {estimator.status_message}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "Positionieren Sie sich korrekt", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
                
                draw_progress_circle(frame, (frame.shape[1] - 80, 80), 50, 0, (100, 100, 100))

            # Anweisungen
            cv2.putText(frame, "SPACE: Start/Reset | Q: Beenden", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow("Age Estimation", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Leertaste
                if estimator.current_status == "ready":
                    estimator.start_processing()
                else:
                    estimator.reset()
                    show_result_until = 0

    finally:
        estimator.reset()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()