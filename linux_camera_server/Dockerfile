FROM almalinux:9

# Grundlegende Systempakete installieren
RUN dnf -y update && \
    dnf -y install epel-release && \
    dnf -y install \
    python3 \
    python3-pip \
    python3-devel \
    gcc \
    make \
    which \
    wget \
    # Für OpenCV benötigte Pakete
    mesa-libGL \
    # Für Multimedia-Unterstützung und Kamerazugriff
    libusb \
    libusbx-devel \
    gstreamer1 \
    gstreamer1-plugins-base \
    && dnf clean all

# Arbeitsverzeichnis erstellen
WORKDIR /app

# Python-Abhängigkeiten installieren
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Anwendungsfiles kopieren
COPY linux_camera_server.py .
COPY start.sh .

# Startskript ausführbar machen
RUN chmod +x start.sh

# Port freigeben
EXPOSE 5002

# Umgebungsvariablen
ENV PYTHONUNBUFFERED=1

# Container beim Start ausführen
CMD ["./start.sh"]
