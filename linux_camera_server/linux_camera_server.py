#!/usr/bin/env python3
"""
Linux Camera Server
----------------
Dieser Server greift auf die Linux-Kamera zu und streamt die Bilder über einen
Flask-Server, damit sie von anderen Anwendungen genutzt werden können.
Performance-optimierte Version für geringe Latenz, angepasst für CentOS.
"""

import cv2
import numpy as np
from flask import Flask, Response, jsonify
import threading
import time
import logging
import argparse
import socket
import queue
import os
import sys

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Globale Variablen
camera = None
is_streaming = False
latest_frame = None
frame_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=2)  # Kleine Queue für aktuelle Frames

def get_local_ip():
    """Ermittelt die lokale IP-Adresse"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"

def capture_frames():
    """Erfasst kontinuierlich Frames von der Kamera"""
    global is_streaming, camera

    logger.info("Capture-Thread gestartet")
    last_frame_time = time.time()
    frames_count = 0

    while is_streaming:
        if camera is None or not camera.isOpened():
            logger.warning("Kamera nicht verfügbar")
            time.sleep(0.5)
            continue

        success, frame = camera.read()
        if success:
            # Frame-Rate berechnen
            now = time.time()
            frames_count += 1
            if now - last_frame_time >= 5.0:  # Alle 5 Sekunden FPS anzeigen
                fps = frames_count / (now - last_frame_time)
                logger.info(f"Kamera-Framerate: {fps:.1f} FPS")
                frames_count = 0
                last_frame_time = now

            # Frame verkleinern für schnellere Übertragung
            frame = cv2.resize(frame, (640, 480))

            # Wenn die Queue voll ist, entferne das älteste Frame
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass

            # Neues Frame in die Queue legen
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass
        else:
            logger.warning("Fehler beim Lesen des Kamera-Frames")

        # Kürzere Verzögerung für höhere Framerate
        time.sleep(0.01)

def generate_frames():
    """Generator für MJPEG-Stream mit optimierter Performance"""
    while True:
        try:
            # Warte max. 0.5 Sekunden auf einen neuen Frame
            frame = frame_queue.get(timeout=0.5)

            # JPEG-Kodierung für das Streaming mit reduzierter Qualität für schnellere Übertragung
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            # Kein Frame verfügbar, warte kurz
            time.sleep(0.05)
            continue
        except Exception as e:
            logger.error(f"Fehler im Stream-Generator: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    """Statusseite"""
    return jsonify({
        "status": "running",
        "streaming": is_streaming
    })

@app.route('/stream')
def stream():
    """MJPEG-Stream-Endpunkt"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/status')
def status():
    """Kamerastatus abfragen"""
    global camera, is_streaming

    if camera is None:
        camera_status = "not_initialized"
    elif not camera.isOpened():
        camera_status = "not_opened"
    else:
        camera_status = "ready"

    return jsonify({
        "camera_status": camera_status,
        "streaming": is_streaming,
        "queue_size": frame_queue.qsize()
    })

def start_camera(camera_index=0):
    """Startet die Kamera mit optimierten Einstellungen für Linux"""
    global camera, is_streaming

    # Prüfen, ob der angegebene Kamerapfad existiert (für v4l2)
    if isinstance(camera_index, str) and camera_index.startswith('/dev/video'):
        if not os.path.exists(camera_index):
            logger.error(f"Kamera-Gerät existiert nicht: {camera_index}")
            # Versuche, verfügbare Kameras zu finden
            available_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
            if available_devices:
                logger.info(f"Verfügbare Kamera-Geräte: {', '.join(available_devices)}")
            return False

    # Unter Linux verschiedene Methoden zum Öffnen der Kamera versuchen
    try:
        # 1. Versuch: Standard OpenCV Öffnung
        camera = cv2.VideoCapture(camera_index)

        # 2. Versuch: Mit explizitem Backend (V4L2 für Linux)
        if not camera.isOpened():
            logger.info(f"Versuche Kamera {camera_index} mit V4L2...")
            camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

        # 3. Versuch: Als String-Pfad zur Gerätedatei
        if not camera.isOpened() and isinstance(camera_index, int):
            device_path = f"/dev/video{camera_index}"
            if os.path.exists(device_path):
                logger.info(f"Versuche Kamera über Pfad: {device_path}...")
                camera = cv2.VideoCapture(device_path)

        if not camera.isOpened():
            logger.error(f"Konnte Kamera {camera_index} nicht öffnen")
            return False

        # Kamera-Einstellungen für bessere Performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Nur neueste Frames behalten

        # Test-Frame
        success, test_frame = camera.read()
        if not success or test_frame is None:
            logger.error("Konnte keinen Test-Frame lesen")
            return False

        logger.info(f"Kamera erfolgreich geöffnet: {test_frame.shape[1]}x{test_frame.shape[0]}")
        is_streaming = True

        # Capture-Thread starten
        capture_thread = threading.Thread(target=capture_frames)
        capture_thread.daemon = True
        capture_thread.start()

        return True

    except Exception as e:
        logger.error(f"Fehler beim Initialisieren der Kamera: {e}")
        return False

def list_available_cameras():
    """Listet alle verfügbaren Kameras auf (Linux-spezifisch)"""
    # Suche nach V4L2-Geräten
    video_devices = []
    try:
        for i in range(10):  # Überprüfe /dev/video0 bis /dev/video9
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path):
                video_devices.append(device_path)
    except Exception as e:
        logger.error(f"Fehler beim Suchen nach Kamera-Geräten: {e}")

    return video_devices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Linux Camera Server')
    parser.add_argument('--port', type=int, default=5002, help='Server-Port (Standard: 5002)')
    parser.add_argument('--camera', type=str, default="0", help='Kamera-Index oder Pfad (Standard: 0)')
    parser.add_argument('--list', action='store_true', help='Verfügbare Kameras auflisten')
    args = parser.parse_args()

    # Verfügbare Kameras auflisten, wenn --list angegeben wurde
    if args.list:
        print("Suche nach verfügbaren Kameras...")
        devices = list_available_cameras()
        if devices:
            print("Gefundene Kamera-Geräte:")
            for device in devices:
                print(f"  - {device}")
        else:
            print("Keine Kamera-Geräte gefunden")
        sys.exit(0)

    # Kamera-Index konvertieren, wenn es eine Zahl ist
    camera_arg = args.camera
    try:
        camera_arg = int(args.camera)
    except ValueError:
        # Bei Fehler: Als Pfad behandeln
        pass

    local_ip = get_local_ip()
    logger.info(f"Starte Linux Camera Server auf IP: {local_ip}, Port: {args.port}")

    if start_camera(camera_arg):
        logger.info(f"Kamera gestartet - Stream verfügbar unter: http://{local_ip}:{args.port}/stream")
        # Threaded=True für bessere Performance
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    else:
        logger.error("Konnte Kamera nicht starten - Server wird nicht gestartet")
