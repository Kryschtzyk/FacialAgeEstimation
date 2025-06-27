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
        self.max_duration = 15.0  # 15 Sekunden maximum
        self.frame_queue = queue.Queue(maxsize=30)  # Größere Queue für mehr Versuche
        self.processing_thread = None
        self.stop_processing = False
        self.last_frame_time = 0
        self.frame_interval = 0.25  # Etwas häufiger versuchen
        self.current_status = "waiting"
        self.status_message = ""
        self.last_valid_check = 0
        self.face_valid = False
        self.total_frames_attempted = 0  # Gesamtanzahl der versuchten Frames
        self.rejected_frames = 0  # Anzahl der abgelehnten Frames
        
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
            
            return True, "Gesicht OK"
            
        except Exception as e:
            return False, f"Fehler: {str(e)[:30]}"
    
    def process_frames_worker(self):
        """
        Worker-Thread für die zeitaufwändige DeepFace-Analyse
        Sammelt so lange bis 8 gültige Frames erreicht sind oder Timeout
        """
        while not self.stop_processing and self.processing:
            try:
                # Prüfe Timeout
                if self.start_time and (time.time() - self.start_time) >= self.max_duration:
                    print(f"Timeout erreicht nach 15s. Gültige Frames: {len(self.valid_ages)}/8")
                    break
                
                # Prüfe ob 8 gültige Frames erreicht
                if len(self.valid_ages) >= 8:
                    print(f"✓ Ziel erreicht: 8/8 gültige Frames gesammelt!")
                    break
                
                # Warte auf neuen Frame
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                    self.total_frames_attempted += 1
                except queue.Empty:
                    continue
                
                # Erst schnelle Validierung
                is_valid_quick, quick_reason = self.detect_face_quick(frame)
                if not is_valid_quick:
                    self.rejected_frames += 1
                    print(f"Frame {self.total_frames_attempted} abgelehnt (schnell): {quick_reason}")
                    continue
                
                # Dann aufwändige DeepFace-Analyse
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
                            self.rejected_frames += 1
                            print(f"Frame {self.total_frames_attempted} abgelehnt (Grimasse): {dominant_emotion} ({emotion_value:.1f}%)")
                            continue
                    
                    # Prüfe auch auf zu starke Freude (übertriebenes Lachen)
                    if dominant_emotion == 'happy' and emotions['happy'] > 85:
                        self.rejected_frames += 1
                        print(f"Frame {self.total_frames_attempted} abgelehnt (übertriebene Freude): {emotions['happy']:.1f}%")
                        continue
                    
                    # Age analysis
                    age_result = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)
                    age = age_result[0]['age']
                    
                    # Validiere Alter (Plausibilitätscheck)
                    if age < 5 or age > 100:
                        self.rejected_frames += 1
                        print(f"Frame {self.total_frames_attempted} abgelehnt (unplausibles Alter): {age:.1f}")
                        continue
                    
                    # ✓ Gültiges Alter hinzufügen
                    self.valid_ages.append(age)
                    valid_count = len(self.valid_ages)
                    print(f"✓ Frame {self.total_frames_attempted} AKZEPTIERT: Alter {age:.1f} ({valid_count}/8 gültige Frames)")
                    
                except Exception as e:
                    self.rejected_frames += 1
                    print(f"Frame {self.total_frames_attempted} abgelehnt (DeepFace Fehler): {e}")
                    continue
                
                # Kurze Pause zwischen Verarbeitungen
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Worker Thread Fehler: {e}")
                time.sleep(0.2)
        
        # Worker Thread beendet - Status aktualisieren
        if len(self.valid_ages) >= 8:
            self.current_status = "complete"
        else:
            self.current_status = "timeout"
        
        self.processing = False
        self.stop_processing = True
    
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
            self.face_valid = True
            self.total_frames_attempted = 0
            self.rejected_frames = 0
            
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
                # Immer den aktuellsten Frame nehmen
                if self.frame_queue.full():
                    # Queue teilweise leeren um Platz zu schaffen
                    try:
                        for _ in range(5):  # Entferne 5 alte Frames
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
            # Kontinuierliche Validierung während der Verarbeitung
            if current_time - self.last_valid_check > 0.2:
                is_valid, reason = self.detect_face_quick(frame)
                self.face_valid = is_valid
                if not is_valid:
                    self.status_message = f"Achtung: {reason}"
                else:
                    valid_count = len(self.valid_ages)
                    remaining_time = max(0, self.max_duration - (current_time - self.start_time))
                    self.status_message = f"Sammle Frames... ({valid_count}/8) - {remaining_time:.1f}s"
                self.last_valid_check = current_time
            
            # Prüfe ob Thread beendet wurde
            if not self.processing_thread.is_alive():
                if len(self.valid_ages) >= 8:
                    self.current_status = "complete"
                    self.status_message = "Analyse erfolgreich abgeschlossen"
                else:
                    self.current_status = "failed"
                    self.status_message = f"Fehler: Nur {len(self.valid_ages)}/8 Frames in 15s"
        
        elif self.current_status in ["complete", "failed", "timeout"]:
            # Status bleibt bis Reset
            pass
        
        else:
            # Prüfe Gesichtsstatus (weniger häufig um Flackern zu vermeiden)
            if current_time - self.last_valid_check > 0.5:
                is_valid, reason = self.detect_face_quick(frame)
                if is_valid:
                    self.current_status = "ready"
                    self.status_message = "Bereit für Analyse"
                else:
                    self.current_status = "waiting"
                    self.status_message = reason
                
                self.last_valid_check = current_time
    
    def get_progress(self):
        """
        Gibt den aktuellen Fortschritt zurück (0.0 bis 1.0)
        """
        if not self.processing and self.current_status not in ["complete", "failed"]:
            return 0.0
        
        if self.current_status == "complete":
            return 1.0
        
        if self.current_status == "failed":
            return min(len(self.valid_ages) / 8.0, 1.0)
        
        # Fortschritt basierend auf gesammelten gültigen Frames
        return len(self.valid_ages) / 8.0
    
    def get_final_age(self):
        """
        Berechnet das finale Alter als Durchschnitt
        """
        if len(self.valid_ages) > 0 and self.final_age is None:
            ages = list(self.valid_ages)
            # Entferne Ausreißer wenn genug Daten vorhanden
            if len(ages) >= 5:
                ages_sorted = sorted(ages)
                ages_sorted = ages_sorted[1:-1]  # Entferne niedrigste und höchste
                self.final_age = int(np.mean(ages_sorted))
            else:
                self.final_age = int(np.mean(ages))
        return self.final_age
    
    def get_frame_count(self):
        return len(self.valid_ages)
    
    def get_attempted_frames(self):
        return self.total_frames_attempted
    
    def get_rejected_frames(self):
        return self.rejected_frames
    
    def get_elapsed_time(self):
        if self.start_time:
            return time.time() - self.start_time
        return 0
    
    def get_remaining_time(self):
        if self.start_time:
            elapsed = time.time() - self.start_time
            return max(0, self.max_duration - elapsed)
        return self.max_duration
    
    def reset(self):
        """
        Setzt den Estimator zurück
        """
        self.stop_processing = True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
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
        self.face_valid = False
        self.total_frames_attempted = 0
        self.rejected_frames = 0


def draw_progress_circle(frame, center, radius, progress, color=(0, 255, 0), face_valid=True):
    """
    Zeichnet einen Fortschrittskreis mit Gesichtsvalidierung
    """
    # Äußerer Kreis - Farbe je nach Gesichtsstatus
    circle_color = color if face_valid else (0, 0, 255)
    cv2.circle(frame, center, radius, (100, 100, 100), 3)
    
    # Fortschrittskreis
    if progress > 0:
        angle = int(360 * progress)
        axes = (radius, radius)
        cv2.ellipse(frame, center, axes, -90, 0, angle, circle_color, 3)
    
    # Innerer Text
    if progress >= 1.0:
        text = "OK"
        color_text = color
    else:
        frames_collected = int(progress * 8)
        text = f"{frames_collected}/8"
        color_text = circle_color
    
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
            
            # Zeige Ergebnis für 6 Sekunden nach Abschluss/Fehler
            if estimator.current_status in ["complete", "failed"] and show_result_until == 0:
                show_result_until = current_time + 6.0
            
            # Reset nach Anzeige des Ergebnisses
            if show_result_until > 0 and current_time > show_result_until:
                estimator.reset()
                show_result_until = 0
            
            # UI basierend auf Status
            if estimator.current_status == "complete":
                final_age = estimator.get_final_age()
                frame_count = estimator.get_frame_count()
                attempted = estimator.get_attempted_frames()
                rejected = estimator.get_rejected_frames()
                
                cv2.putText(frame, f"Geschätztes Alter: {final_age} Jahre", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Basierend auf {frame_count} gültigen Frames", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"({attempted} Frames getestet, {rejected} abgelehnt)", 
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
                
                draw_progress_circle(frame, (frame.shape[1] - 80, 80), 50, 1.0, (0, 255, 0), True)
            
            elif estimator.current_status == "failed":
                frame_count = estimator.get_frame_count()
                attempted = estimator.get_attempted_frames()
                rejected = estimator.get_rejected_frames()
                
                cv2.putText(frame, "Alter konnte nicht ermittelt werden", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Unzureichende Frames: {frame_count}/8 in 15s", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f"({attempted} getestet, {rejected} abgelehnt)", 
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
                cv2.putText(frame, "Versuchen Sie es erneut mit besserem Licht/Position", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                
                progress = estimator.get_progress()
                draw_progress_circle(frame, (frame.shape[1] - 80, 80), 50, progress, (0, 0, 255), False)
            
            elif estimator.current_status == "processing":
                progress = estimator.get_progress()
                frame_count = estimator.get_frame_count()
                remaining = estimator.get_remaining_time()
                attempted = estimator.get_attempted_frames()
                rejected = estimator.get_rejected_frames()
                
                # Fortschrittskreis - rot wenn Gesicht nicht gültig
                draw_progress_circle(frame, (frame.shape[1] - 80, 80), 50, progress, 
                                   (0, 255, 255), estimator.face_valid)
                
                # Status-Text
                text_color = (0, 255, 255) if estimator.face_valid else (0, 100, 255)
                cv2.putText(frame, f"Sammle 8 gültige Frames... ({frame_count}/8)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                
                # Detailstatus
                status_color = (0, 255, 255) if estimator.face_valid else (0, 0, 255)
                cv2.putText(frame, estimator.status_message, 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                
                cv2.putText(frame, f"Zeit: {remaining:.1f}s verbleibend", 
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                
                if attempted > 0:
                    success_rate = ((attempted - rejected) / attempted) * 100
                    cv2.putText(frame, f"Erfolgsrate: {success_rate:.0f}% ({attempted-rejected}/{attempted})", 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Füge Frame zur Verarbeitung hinzu (nur wenn aktuell gültig)
                if estimator.face_valid:
                    estimator.add_frame_for_processing(frame)
            
            elif estimator.current_status == "ready":
                cv2.putText(frame, "Bereit für Analyse!", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "Drücken Sie SPACE zum Starten", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, "8 gültige Frames werden gesammelt (max 15s)", 
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
                
                draw_progress_circle(frame, (frame.shape[1] - 80, 80), 50, 0, (0, 255, 0), True)
            
            else:  # waiting
                cv2.putText(frame, f"Warten: {estimator.status_message}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "Positionieren Sie sich korrekt", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
                
                draw_progress_circle(frame, (frame.shape[1] - 80, 80), 50, 0, (100, 100, 100), False)

            # Anweisungen
            cv2.putText(frame, "SPACE: Start/Reset | Q: Beenden", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow("Age Estimation", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
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