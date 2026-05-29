from fastapi import FastAPI,HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import asyncio
from colored_print import log

from file_handler import FileHandler
from rag_logic import RAGLogic
from audio_manager import AudioManager


MODEL_NAME = "mistral:7b"  # Nom du modèle Ollama à utiliser

class QuestionRequest(BaseModel):
    question: str

file_handler = FileHandler()
asyncio.run(file_handler.init_db())  # Initialisation de la base de données au démarrage de l'application

logic = RAGLogic(file_handler=file_handler)  # Création de l'instance de RAGLogic avec le FileHandler initialisé

app = FastAPI()
audio_manager = None
try:
    audio_manager = AudioManager()
except Exception as e:
    print(f"Erreur lors de l'initialisation du gestionnaire audio : {e}")

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Permettre les requêtes de n'importe quelle origine (à ajuster en production)
    allow_methods=["*"],  # Permettre toutes les méthodes HTTP
    allow_headers=["*"],  # Permettre tous les en-têtes
    allow_credentials=False,  # Permettre l'envoi de cookies et d'informations d'identification
)

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de RAG local!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/history")
async def get_history():
    return { "chat_history" : logic.memory}

@app.post("/load_files")
async def load_files():
    # Logique pour charger les fichiers PDF et les ajouter à la base de données
    pass

@app.post("/ask")
async def ask_rag(question: QuestionRequest):
    try:
        answer = await logic.ask(question.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if audio_manager is None:
        raise HTTPException(status_code=500, detail="Le gestionnaire audio n'est pas disponible. Assurez-vous que le modèle Vosk est correctement installé.")
    log.warn(f"Fichier audio reçu pour transcription: {file.filename}")
    audio_file_path = f"temp_{file.filename}"
    with open(audio_file_path, "wb") as f:
        f.write(await file.read())
    transcription = await audio_manager.transcribe_audio(audio_file_path)
    log.success(f"Transcription obtenue : {transcription}")
    return {"transcription": transcription}