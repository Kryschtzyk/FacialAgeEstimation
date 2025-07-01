FROM python:3.9-slim

# System-Abhängigkeiten installieren
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    && rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis setzen
WORKDIR /app

# Python-Abhängigkeiten kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Anwendungscode kopieren
COPY . .

# Logs-Verzeichnis erstellen
RUN mkdir -p /app/logs

# Port freigeben
EXPOSE 5001

# Umgebungsvariablen
ENV PYTHONPATH=/app
ENV FLASK_ENV=production

# Startkommando
CMD ["python", "app_websocket.py"]
