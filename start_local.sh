#!/bin/bash

# Lokaler Start des Altersschätzungs-Systems
# Dieses Script startet das System lokal ohne Docker für besseren Kamera-Zugriff

echo "🎥 Starte lokales Altersschätzungs-System..."

# Prüfe ob Python Virtual Environment existiert
if [ ! -d ".venv" ]; then
    echo "📦 Erstelle Python Virtual Environment..."
    python3 -m venv .venv
fi

# Aktiviere Virtual Environment
echo "🔧 Aktiviere Virtual Environment..."
source .venv/bin/activate

# Installiere Abhängigkeiten
echo "📥 Installiere Python-Pakete..."
pip install --upgrade pip
pip install -r requirements.txt

# Erstelle Logs-Verzeichnis
mkdir -p logs

# Prüfe Kamera-Zugriff
echo "📹 Prüfe Kamera-Zugriff..."
python3 -c "
import cv2
import sys

try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print('✅ Kamera erfolgreich gefunden und getestet')
            print(f'   Auflösung: {frame.shape[1]}x{frame.shape[0]}')
        else:
            print('❌ Kamera gefunden, aber kein Frame erhalten')
            sys.exit(1)
        cap.release()
    else:
        print('❌ Keine Kamera gefunden oder Zugriff verweigert')
        sys.exit(1)
except Exception as e:
    print(f'❌ Kamera-Fehler: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🚀 Starte WebSocket-Server..."
    echo "📱 Kiosk-Interface: http://localhost:5001"
    echo "🖥️  Admin-Interface: http://localhost:5001/admin"
    echo ""
    echo "Drücken Sie Ctrl+C zum Beenden"
    echo ""

    # Starte den WebSocket-Server
    python3 app_websocket.py
else
    echo ""
    echo "❌ Kamera-Test fehlgeschlagen. Bitte prüfen Sie:"
    echo "   1. Ist eine Kamera angeschlossen?"
    echo "   2. Haben andere Programme Zugriff auf die Kamera?"
    echo "   3. Sind die Kamera-Berechtigungen gesetzt?"
    echo ""
    echo "💡 Auf macOS: Systemeinstellungen > Sicherheit > Kamera"
fi
