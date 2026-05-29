#!/bin/bash

# Start Ollama in the background
ollama serve &

# Wait for Ollama to wake up
echo "Waiting for Ollama server to start..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 1
done

echo "Ollama server is up and running."
ollama list

# Pull the model
if ! ollama list | grep -q "mistral"; then
    echo "Pulling mistral:7b..."
    ollama pull mistral:7b
    echo "Model pulled successfully."
else
    echo "Model mistral:7b already exists. Skipping pull."
fi

# Start your Python application
echo "Starting Uvicorn..."
exec uvicorn srcs.main:app --host 0.0.0.0 --port 8000