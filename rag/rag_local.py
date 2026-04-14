from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_classic.chains import RetrievalQA
import os
import json
import speech_recognition as sr
import pyaudio
from colored_print import log
from vosk import Model, KaldiRecognizer
import asyncio

model = Model(model_name="vosk-model-small-fr-0.22")
rec = KaldiRecognizer(model, 16000)

pa = pyaudio.PyAudio()

mic_setup = False

for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    print(f"Index {i}: {info['name']} - Entrées: {info['maxInputChannels']}")

print("Veuillez entrer l'index du périphérique d'entrée audio que vous souhaitez utiliser : ")

input_index = int(input())
r = sr.Recognizer()
r.energy_threshold = 100

mic = sr.Microphone(device_index=input_index)

try:
    with mic as source:
        r.adjust_for_ambient_noise(source)
        log.info("Test de reconnaissance vocale en cours. Parlez maintenant...")
        audio = r.listen(source, timeout=5)
        try:
            log.success(f"Test réussi!")
            mic_setup = True
        except sr.WaitTimeoutError:
            log.warn("Délai d'attente dépassé lors du test de reconnaissance vocale.")
except Exception as e:
    log.err(f"Erreur lors de l'accès au microphone : {e}")

DB_DIR = "./chroma_db"
FILES_DIR = "./rag/db/"
FILES = os.listdir(FILES_DIR)
FILES_PATHS = [os.path.join(FILES_DIR, file) for file in FILES]
EMBEDDING = OllamaEmbeddings(model="mistral")
COLLECTION = "rag_collection"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
collection = None

async def load_file(file_path):
    log.info(f"Chargement du fichier : {file_path}")
    ext = file_path.split(".")[-1].lower()
    loader = None
    if ext == "pdf":
        try:
            loader = PyPDFLoader(file_path=file_path, mode="single")
        except Exception as e:
            log.err(f"Erreur lors du chargement du fichier {file_path} : {e}")
    elif ext == "docx":
        try:
            loader = Docx2txtLoader(file_path=file_path)
        except Exception as e:
            log.err(f"Erreur lors du chargement du fichier {file_path} : {e}")
    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return [{"page_content": content, "metadata": {"source": file_path}}]
    else:
        log.err(f"Type de fichier non supporté : {ext}")

    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, loader.load)
    return data

async def init_db():
    if not os.path.exists(DB_DIR):
        log.warn("Le répertoire de la base de données n'existe pas. Création du répertoire...")
        if not os.path.exists(FILES_DIR):
            log.err("Le répertoire 'rag/db/' est introuvable. Veuillez vous assurer qu'il est présent dans le répertoire de travail.")
            exit(1)

        documents = []
        log.info(f"Chargement des fichiers PDF depuis le répertoire '{FILES_DIR}' cela peut prendre un peu de temps...")
        for file_path in FILES_PATHS:
            log.info(f"Fichier trouvé : {file_path}")
            data = await load_file(file_path)
            documents.extend(data)

        chunks = text_splitter.split_documents(documents)

        collection = Chroma.from_documents(
            documents=chunks,
            embedding=EMBEDDING,
            persist_directory=DB_DIR,
            collection_name=COLLECTION
        )
    else:
        log.info("Le répertoire de la base de données existe déjà.")
        log.info("Vérification des fichiers présents dans la collection...")
            
        collection = Chroma(
            embedding_function=EMBEDDING,
            persist_directory=DB_DIR,
            collection_name=COLLECTION
        )
        
        res = collection.get(include=["metadatas", "documents"])
        metadatas = [meta.get('source') for meta, doc in zip(res['metadatas'], res['documents'])]
        metadatas = list(set(metadatas))
        for file_path in FILES_PATHS:
            if not file_path in metadatas:
                log.warn(f"Le fichier '{file_path}' n'est pas présent dans la collection. Ajout en cours...")
                data = await load_file(file_path)
                chunks = text_splitter.split_documents(data)
                for chunk in chunks:
                    chunk.metadata['source'] = file_path
                
                collection.add_documents(documents=chunks)

    log.success("Collection chargée avec succès!")
    return collection

if __name__ == "__main__":
    collection = asyncio.run(init_db())
    # Après avoir chargé la collection
    res = collection.get(include=["metadatas", "documents"])
    metadatas = [meta.get('source') for meta, doc in zip(res['metadatas'], res['documents'])]
    metadatas = list(set(metadatas))
    print(f"Fichiers présents dans la collection : {metadatas}")
    
    retriever = collection.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.25, "score_threshold": 0.2}
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=OllamaLLM(model="mistral"),
        chain_type="stuff",
        retriever=retriever
    )
    
    while 1:
        if mic_setup:
            log.info("Vous pouvez poser une question en tapant ou en utilisant la commande 'listen' pour parler.")
        else:
            log.warn("Le microphone n'est pas configuré. Veuillez poser votre question en tapant.")
        demand = input("Entrez votre question (ou 'exit' pour quitter) : ")
        if demand.lower() == 'exit':
            break
        elif demand.lower() == 'listen' and mic_setup:
            with mic as source:
                r.adjust_for_ambient_noise(source)
                log.info("Parlez maintenant...")
                try:
                    audio = r.listen(source, timeout=5)
                    rec.AcceptWaveform(audio.get_raw_data(convert_rate=16000, convert_width=2))
                    res = json.loads(rec.Result())
                    try:
                        valid = input(f"Vous avez dit : '{res['text']}'?\n(Enter: Y/<Ctrl-C>: N) : ")
                        response = qa_chain.invoke(res['text'])
                        print(f'Reponse:\n {response["result"]}')
                    except KeyboardInterrupt:
                        log.info("Opération annulée par l'utilisateur.")
                        continue
                except sr.UnknownValueError:
                    log.warn("Désolé, je n'ai pas compris. Veuillez réessayer.")
                    continue
                except sr.WaitTimeoutError:
                    log.warn("Délai d'attente dépassé.")
                    continue
        else:
            response = qa_chain.invoke(demand)
            print(f'Reponse:\n {response["result"]}')
            
    pa.terminate()
