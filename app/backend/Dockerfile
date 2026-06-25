FROM ollama/ollama:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 \
    curl zstd gcc ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY srcs/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-multipart
RUN python3 -m spacy download fr_core_news_sm

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY ./vosk-model-small-fr-0.22 /app/vosk-model/vosk-model-small-fr-0.22
COPY srcs/ /app/srcs/

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
