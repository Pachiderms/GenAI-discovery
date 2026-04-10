from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
import os
import speech_recognition as sr
import pyaudio
from colored_print import log

pa = pyaudio.PyAudio()

mic_setup = False

for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    print(f"Index {i}: {info['name']} - Entrées: {info['maxInputChannels']}")

print("Veuillez entrer l'index du périphérique d'entrée audio que vous souhaitez utiliser : ")

input_index = int(input())
r = sr.Recognizer()
r.energy_threshold = 300

m = sr.Microphone(device_index=input_index)

try:
    with m as source:
        r.adjust_for_ambient_noise(source)
        log.info("Test de reconnaissance vocale en cours. Parlez maintenant...")
        audio = r.listen(source, timeout=5)
        try:
            test_text = r.recognize_google(audio, language="fr-FR")
            log.success(f"Test réussi! Vous avez dit : '{test_text}'")
            mic_setup = True
        except sr.WaitTimeoutError:
            log.warn("Délai d'attente dépassé lors du test de reconnaissance vocale.")
except Exception as e:
    log.err(f"Erreur lors de l'accès au microphone : {e}")

DB_DIR = "./chroma_db"
FILES_DIR = "./rag/db/"
FILES = os.listdir(FILES_DIR)
EMBEDDING = OllamaEmbeddings(model="mistral:7b")
COLLECTION = "rag_collection"
collection = None
if not os.path.exists(DB_DIR):
    log.warn("Le répertoire de la base de données n'existe pas. Création du répertoire...")
    if not os.path.exists(FILES_DIR):
        log.err("Le répertoire 'rag/db/' est introuvable. Veuillez vous assurer qu'il est présent dans le répertoire de travail.")
        exit(1)

    documents = []
    for file in FILES:
        log.info(f"Fichier trouvé : {file}")
        loader = PyPDFLoader(file_path=os.path.join(FILES_DIR, file), mode="single")
        data = loader.load()
        documents.extend(data)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    collection = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDING,
        persist_directory=DB_DIR,
        collection_name=COLLECTION
    )
else:
    log.info("Le répertoire de la base de données existe déjà. Chargement de la collection existante...")
    collection = Chroma(
        embedding_function=EMBEDDING,
        persist_directory=DB_DIR,
        collection_name=COLLECTION
    )

log.success("Collection chargée avec succès!")

qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="mistral:7b"),
    chain_type="stuff",
    retriever=collection.as_retriever()
)

if __name__ == "__main__":
    while 1:
        question = None
        if mic_setup:
            log.info("Vous pouvez poser une question en tapant ou en utilisant la commande 'listen' pour parler.")
        else:
            log.warn("Le microphone n'est pas configuré. Veuillez poser votre question en tapant.")
        demand = input("Entrez votre question (ou 'exit' pour quitter) : ")
        if demand.lower() == 'exit':
            break
        elif demand.lower() == 'listen' and mic_setup:
            log.info("Écoute en cours...")
            with m as source:
                r.adjust_for_ambient_noise(source)
                log.info("Parlez maintenant...")
                audio = r.listen(source, timeout=5)
                try:
                    question = r.recognize_google(audio, language="fr-FR")
                    res = input(f"Vous avez dit : '{question}'?\n(O/N) : ")
                    if res.lower() != "oui" and res.lower() != "o":
                        question = None
                        continue
                    else:
                        response = qa_chain.invoke(question)
                        print(f'Reponse:\n {response["result"]}')
                except sr.UnknownValueError:
                    log.warn("Désolé, je n'ai pas compris. Veuillez réessayer.")
                    continue
                except sr.WaitTimeoutError:
                    log.warn("Délai d'attente dépassé.")
                    continue
        else:
            question = demand
            response = qa_chain.invoke(question)
            print(f'Reponse:\n {response["result"]}')
            
pa.terminate()
