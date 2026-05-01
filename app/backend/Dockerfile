FROM python:3.10-slim

# Install dependencies for audio and system
RUN apt-get update && apt-get install -y \
    libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 \
    gcc ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les modèles Vosk dans le conteneur
COPY ./vosk-model-small-fr-0.22 /app/vosk-model

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]