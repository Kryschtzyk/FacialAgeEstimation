#!/usr/bin/env python3
"""
Mac Camera Server
----------------
Dieser Server greift auf die Mac-Kamera zu und streamt die Bilder über einen
Flask-Server, damit sie vom Docker-Container genutzt werden können.
"""

import cv2
import numpy as np
from flask import Flask, Response, jsonify
import threading
import time
import logging
import argparse
import socket

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
    global latest_frame, is_streaming, camera

    while is_streaming:
        if camera is None or not camera.isOpened():
            logger.warning("Kamera nicht verfügbar")
            time.sleep(1)
            continue

        success, frame = camera.read()
        if success:
            with frame_lock:
                latest_frame = frame.copy()
        else:
            logger.warning("Fehler beim Lesen des Kamera-Frames")

        time.sleep(1/30)  # 30 FPS

def generate_frames():
    """Generator für MJPEG-Stream"""
    global latest_frame

    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.1)
                continue

            # JPEG-Kodierung für das Streaming
            _, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(1/30)  # 30 FPS

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
        "streaming": is_streaming
    })

def start_camera(camera_index=0):
    """Startet die Kamera"""
    global camera, is_streaming

    # Versuche verschiedene Backends für Mac
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        logger.info(f"Versuche Kamera {camera_index} mit AVFoundation...")
        camera = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

    if not camera.isOpened():
        logger.error(f"Konnte Kamera {camera_index} nicht öffnen")
        return False

    # Kamera-Einstellungen
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mac Camera Server')
    parser.add_argument('--port', type=int, default=5002, help='Server-Port (Standard: 5002)')
    parser.add_argument('--camera', type=int, default=0, help='Kamera-Index (Standard: 0)')
    args = parser.parse_args()

    local_ip = get_local_ip()
    logger.info(f"Starte Mac Camera Server auf IP: {local_ip}, Port: {args.port}")

    if start_camera(args.camera):
        logger.info(f"Kamera gestartet - Stream verfügbar unter: http://{local_ip}:{args.port}/stream")
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    else:
        logger.error("Konnte Kamera nicht starten - Server wird nicht gestartet")
