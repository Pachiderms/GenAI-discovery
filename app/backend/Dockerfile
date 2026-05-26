FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/srcs

RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 \
    curl zstd gcc ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY srcs/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install python-multipart

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY ./vosk-model-small-fr-0.22 /app/vosk-model
COPY srcs/ /app/srcs/

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

CMD ["/entrypoint.sh"]