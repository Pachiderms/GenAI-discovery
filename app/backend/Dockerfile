FROM ollama/ollama:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip \
    libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 \
    curl zstd gcc ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/srcs

WORKDIR /app

COPY srcs/requirements.txt ./
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt && pip install --break-system-packages python-multipart

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY ./vosk-model-small-fr-0.22 /app/vosk-model/vosk-model-small-fr-0.22
COPY srcs/ /app/srcs/

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]

#CMD ["/entrypoint.sh"]