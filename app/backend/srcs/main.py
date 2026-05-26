from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import asyncio
from file_handler import FileHandler
from rag_logic import RAGLogic


class QuestionRequest(BaseModel):
    question: str

file_handler = FileHandler()
asyncio.run(file_handler.init_db())  # Initialisation de la base de données au démarrage de l'application

logic = RAGLogic(file_handler=file_handler)  # Création de l'instance de RAGLogic avec le FileHandler initialisé

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de RAG local!"}

@app.get("/history")
async def get_history():
    # Logique pour récupérer l'historique des conversations
    pass

@app.post("/load_files")
async def load_files():
    # Logique pour charger les fichiers PDF et les ajouter à la base de données
    pass

@app.get("/ask/{qst}")
async def ask_rag(question: QuestionRequest | None = None,
    qst: str | None = None):
    # Logique qa_chain
    if question is None and qst is not None:
        question = QuestionRequest(question=qst)
    elif question is None and qst is None:
        return {"error": "Aucune question fournie"}
    answer = await logic.ask(question.question)
    return {"answer": answer}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Logique Vosk pour transformer le fichier audio reçu en texte
    # logique RAG
    return {"text": "Texte transcrit"}