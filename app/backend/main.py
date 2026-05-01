from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
# ... (tes imports LangChain et Vosk)

app = FastAPI()

class Question(BaseModel):
    text: str

@app.post("/ask")
async def ask_rag(question: Question):
    # La logique de ton qa_chain.invoke
    response = qa_chain.invoke(question.text)
    return {"response": response["result"]}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Logique Vosk pour transformer le fichier audio reçu en texte
    # Puis tu peux appeler le RAG
    return {"text": "Texte transcrit"}