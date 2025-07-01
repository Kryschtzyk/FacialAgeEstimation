#!/bin/bash
# Start-Skript für den Linux-Camera-Server

# Ausgabe in Echtzeit (ohne Puffer)
export PYTHONUNBUFFERED=1

echo "Suche nach verfügbaren Kameras..."
python3 linux_camera_server.py --list

echo ""
echo "Starte Camera Server..."
# Standardmäßig auf Port 5002, kann über Umgebungsvariable überschrieben werden
CAMERA_PORT=${CAMERA_PORT:-5002}
CAMERA_DEVICE=${CAMERA_DEVICE:-0}

echo "Verwende Kamera: $CAMERA_DEVICE auf Port: $CAMERA_PORT"
python3 linux_camera_server.py --port $CAMERA_PORT --camera $CAMERA_DEVICE
