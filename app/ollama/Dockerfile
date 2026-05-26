FROM ollama/ollama
RUN ollama serve & sleep 5 && ollama pull mistral:7b && pkill ollama
CMD ["ollama", "serve"]