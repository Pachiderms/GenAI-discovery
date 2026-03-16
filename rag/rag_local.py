from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
import os

DB_DIR = "./chroma_db"
FILE_PATH = "./rag/rag_test.pdf"
EMBEDDING = OllamaEmbeddings(model="mistral")
COLLECTION = "rag_collection"
collection = None
if not os.path.exists(DB_DIR):
    print("Le répertoire de la base de données n'existe pas. Création du répertoire...")
    if not os.path.exists(FILE_PATH):
        print("Le fichier 'rag_test.pdf' est introuvable. Veuillez vous assurer qu'il est présent dans le répertoire de travail.")
        exit(1)

    loader = PyPDFLoader(file_path=FILE_PATH, mode="single")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)

    collection = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDING,
        persist_directory=DB_DIR,
        collection_name=COLLECTION
    )
else:
    print("Le répertoire de la base de données existe déjà. Chargement de la collection existante...")
    collection = Chroma(
        embedding_function=EMBEDDING,
        persist_directory=DB_DIR,
        collection_name=COLLECTION
    )

qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="mistral"),
    chain_type="stuff",
    retriever=collection.as_retriever()
)

if __name__ == "__main__":
    while 1:
        question = input("Posez votre question (ou tapez 'exit' pour quitter) : ")
        if question.lower() == 'exit':
            break
        response = qa_chain.invoke(question)
        print(f'Reponse:\n {response["result"]}')
