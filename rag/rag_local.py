from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
import os

if not os.path.exists("./rag/rag_test.pdf"):
    print("Le fichier 'rag_test.pdf' est introuvable. Veuillez vous assurer qu'il est présent dans le répertoire de travail.")
    exit(1)
loader = PyPDFLoader(file_path="./rag/rag_test.pdf",
                     mode="single"
                     )
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=OllamaEmbeddings(model="mistral")
)

qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="mistral"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

if __name__ == "__main__":
    while 1:
        question = input("Posez votre question (ou tapez 'exit' pour quitter) : ")
        if question.lower() == 'exit':
            break
        response = qa_chain.invoke(question)
        print(f'Reponse:\n {response["result"]}')
