from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
# ... (tes imports LangChain et Vosk)

app = FastAPI()

class Question(BaseModel):
    text: str

@app.post("/load_files")
async def load_files():
    # Logique pour charger les fichiers PDF et les ajouter à la base de données


@app.post("/ask")
async def ask_rag(question: Question):
    # La logique de ton qa_chain.invoke
    return {"answer": "Réponse du RAG"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Logique Vosk pour transformer le fichier audio reçu en texte
    # Puis tu peux appeler le RAG
    return {"text": "Texte transcrit"}