from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain import RetrievalQA

# 1. Charger le document PDF
loader = PyPDFLoader("mon_document.pdf")
data = loader.load()

# 2. Découper le texte en morceaux digestes (Chunks)
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

question = "Quel est le sujet principal de ce document ?"
print(qa_chain.invoke(question))