from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_classic.chains import RetrievalQA
import os
import json
import speech_recognition as sr
import pyaudio
from vosk import Model, KaldiRecognizer
import asyncio
import spacy
from prompt import system_prompt, nlp_prompt


class bcolors:
    BLACK = '\033[0;30m'
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'	
    YELLOW = '\033[0;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[0;37m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


model = None
rec = None
pa = None
mic_setup = False
r = None


try:
    input_index = int(input(bcolors.PURPLE + "0 to skip Microphone setup\n" + bcolors.WHITE))
    if input_index == 0:
        raise Exception("Skipping Microphone setup...")
    
    model = Model(model_name="vosk-model-small-fr-0.22")
    KaldiRecognizer(model, 16000)
    pa = pyaudio.PyAudio()

    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(f"Index {i}: {info['name']} - Entrées: {info['maxInputChannels']}")

    print("Veuillez entrer l'index du périphérique d'entrée audio que vous souhaitez utiliser : ")

    input_index = int(input())
    r = sr.Recognizer()
    r.energy_threshold = 100

    mic = sr.Microphone(device_index=input_index)

    with mic as source:
        r.adjust_for_ambient_noise(source)
        print(bcolors.CYAN + "Test de reconnaissance vocale en cours. Parlez maintenant...")
        audio = r.listen(source, timeout=5)
        try:
            print(bcolors.GREEN + f"Test réussi!")
            mic_setup = True
        except sr.WaitTimeoutError:
            print("Délai d'attente dépassé lors du test de reconnaissance vocale.")
except Exception as e:
    print(bcolors.YELLOW + f"Erreur lors de l'accès au microphone : {e}\n Le Microphone sera inutilisable durant la session.")


DB_DIR = "./chroma_db"
FILES_DIR = "/db/"
DATA = os.listdir(FILES_DIR)
# print(DATA)
DIRS = [what for what in DATA if os.path.isdir(what) == False]
# print(f"{DIRS=}")
files = [what for what in DATA if os.path.isdir(what) == True]
# print(f"{files=}")
DIRS_PATHS = [os.path.join(FILES_DIR, dir) for dir in DIRS]
for dir in DIRS_PATHS:
    files.extend([os.path.join(dir, file) for file in os.listdir(dir)])
# print(f"after extend: {files=}")
EMBEDDING = OllamaEmbeddings(model="nomic-embed-text")
COLLECTION = "batcave"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
collection = None


async def load_file(file_path):
    print(bcolors.YELLOW + f"Chargement du fichier : {file_path}")
    ext = file_path.split(".")[-1].lower()
    loader = None
    if ext == "pdf":
        try:
            loader = PyPDFLoader(file_path=file_path, mode="single")
        except Exception as e:
            print(bcolors.RED +  f"Erreur lors du chargement du fichier {file_path} : {e}")
    elif ext == "docx":
        try:
            loader = Docx2txtLoader(file_path=file_path)
        except Exception as e:
            print(bcolors.RED +  f"Erreur lors du chargement du fichier {file_path} : {e}")
    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return [{"page_content": content, "metadata": {"source": file_path}}]
    else:
        print(bcolors.RED + f"Type de fichier non supporté : {ext}")

    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, loader.load)
    return data

ollama_model = OllamaLLM(model="mistral")
history = []
nlp = spacy.load("fr_core_news_sm")


async def init_db():
    if not os.path.exists(DB_DIR):
        print(bcolors.CYAN + "Le répertoire de la base de données n'existe pas. Création du répertoire...")
        if not os.path.exists(FILES_DIR):
            print(bcolors.RED + "Le répertoire 'rag/db/' est introuvable. Veuillez vous assurer qu'il est présent dans le répertoire de travail.")
            exit(1)

        documents = []
        print(bcolors.CYAN + f"Chargement des fichiers PDF depuis le répertoire '{FILES_DIR}' cela peut prendre un peu de temps...")
        for file_path in files:
            print(f"Fichier trouvé : {file_path}")
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
        print(bcolors.CYAN + "Le répertoire de la base de données existe déjà.")
        print(bcolors.CYAN + "Vérification des fichiers présents dans la collection...")
            
        collection = Chroma(
            embedding_function=EMBEDDING,
            persist_directory=DB_DIR,
            collection_name=COLLECTION
        )
        
        res = collection.get(include=["metadatas", "documents"])
        metadatas = [meta.get('source') for meta, doc in zip(res['metadatas'], res['documents'])]
        metadatas = list(set(metadatas))
        for file_path in files:
            if not file_path in metadatas:
                print(bcolors.CYAN + f"Le fichier '{file_path}' n'est pas présent dans la collection. Ajout en cours...")
                data = await load_file(file_path)
                chunks = text_splitter.split_documents(data)
                for chunk in chunks:
                    doc_nlp = nlp(chunk.page_content)

                    people = []
                    dates = []
                    orgs = []
                    gpe = []
                    loc =[]
                    product = []
                    event = []

                    for ent in doc_nlp.ents:

                        if ent.label_ == "PER":
                            people.append(ent.text)
                        elif ent.label_ == "DATE":
                            dates.append(ent.text)
                        elif ent.label_ == "ORG":
                            orgs.append(ent.text)
                        elif ent.label_ == "GPE":
                            gpe.append(ent.text)
                        elif ent.label_ == "LOC":
                            loc.append(ent.text)
                        elif ent.label_ == "PRODUCT":
                            product.append(ent.text)
                        elif ent.label_ == "EVENT":
                            event.append(ent.text)

                    chunk.metadata = {
                        "source": file_path,
                        "people": list(set(people)),
                        "dates": list(set(dates)),
                        "organizations": list(set(orgs)),
                        "geopolictical_entity": list(set(gpe)),
                        "location": list(set(loc)),
                        "product": list(set(product)),
                        "event": list(set(event)),
                    }
                
                collection.add_documents(documents=chunks)

    print(bcolors.GREEN + "Collection chargée avec succès!")
    return collection


collection = asyncio.run(init_db())
# Après avoir chargé la collection
res = collection.get(include=["metadatas", "documents"])
metadatas = [meta.get('source') for meta, doc in zip(res['metadatas'], res['documents'])]
metadatas = list(set(metadatas))
print(f"Fichiers présents dans la collection : {len(metadatas)}")
    
retriever = collection.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.25, "score_threshold": 0.2},
    temperature=0,
)

qa_chain = RetrievalQA.from_chain_type(
        llm=ollama_model,
        chain_type="stuff",
        retriever=retriever
)


def ner_score(doc, entities):
    score = 0

    for ent in entities:
        txt = ent["text"]

        if txt in doc.metadata.get("people", []):
            score += 5

        if txt in doc.metadata.get("organizations", []):
            score += 5
            
        if txt in doc.metadata.get("location", []):
            score += 5
            
        if txt in doc.metadata.get("product", []):
            score += 5
            
        if txt in doc.metadata.get("event", []):
            score += 5

        if txt in doc.metadata.get("dates", []):
            score += 3
            
        if txt in doc.metadata.get("geopolictical_entity", []):
            score += 3

    return score


def recognition(query):
    doc = nlp(query)
    
    return [
        {
            'text': ent.text,
            'label': ent.label_
        }
        for ent in doc.ents
    ]


def preproc_query(query):
    
    augmented_query = json.loads(ollama_model.invoke(nlp_prompt.format(question=query)))
    print(bcolors.CYAN + f"{augmented_query=}")
    
    
    new_query = augmented_query['response']
    docs = retriever.invoke(new_query)
    entities = recognition(new_query)
    
    docs = sorted(
        docs,
        key=lambda d: ner_score(d, entities),
        reverse=True
    )
    
    docs = docs[:5]
    
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )
    
    final_query = system_prompt.format(history=history[-5:], context=context, question=new_query)
    
    response = ollama_model.invoke(final_query)
    
    history.append({
        'question': new_query,
        'response': response
    })
    
    # print(f"{context=}\n {history=}")
    
    return response
    

def main():    
    while 1:
        if mic_setup:
            print(bcolors.CYAN + "Vous pouvez poser une question en tapant ou en utilisant la commande 'listen' pour parler.")
        else:
            print(bcolors.YELLOW + "Le microphone n'est pas configuré.\n" + "Veuillez poser votre question en tapant.")
        demand = input(bcolors.PURPLE + "Entrez votre question (ou 'exit' pour quitter) : " + bcolors.WHITE)
        if demand.lower() == 'exit':
            break
        elif demand.lower() == 'listen' and mic_setup:
            with mic as source:
                r.adjust_for_ambient_noise(source)
                print(bcolors.PURPLE + "Parlez maintenant...")
                try:
                    audio = r.listen(source, timeout=5)
                    rec.AcceptWaveform(audio.get_raw_data(convert_rate=16000, convert_width=2))
                    res = json.loads(rec.Result())
                    try:
                        input(f"Vous avez dit : '{res['text']}'?\n(Enter: Y/<Ctrl-C>: N) : ")
                        print(f'Reponse:\n {preproc_query(res['text'])}')
                    except KeyboardInterrupt:
                        print(bcolors.CYAN + "Opération annulée par l'utilisateur.")
                        continue
                except sr.UnknownValueError:
                    print(bcolors.PURPLE + "Désolé, je n'ai pas compris. Veuillez réessayer.")
                    continue
                except sr.WaitTimeoutError:
                    print(bcolors.PURPLE + "Délai d'attente dépassé.")
                    continue
        else:
            print(bcolors.BLUE + bcolors.BOLD + f'Reponse:\n {preproc_query(demand)}')

    if mic_setup:            
        pa.terminate()


if __name__ == "__main__":
    main()