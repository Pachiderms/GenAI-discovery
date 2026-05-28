import os
import asyncio
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from colored_print import log

class FileHandler:
    def __init__(self):
        base_dir = os.getcwd()
        self.files_dir = os.path.join(base_dir, "db") if os.path.isdir(os.path.join(base_dir, "db")) else None
        self.db_dir = os.path.join(base_dir, "chroma_db")
        print(f"Répertoire des fichiers : {self.files_dir}")
        print(f"Répertoire de la base de données : {self.db_dir}")
        self.files = os.listdir(self.files_dir) if self.files_dir and os.path.isdir(self.files_dir) else []
        self.embedding = OllamaEmbeddings(model="mistral:7b")
        self.collection_name = "ultron"
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        self.collection = Chroma(
            embedding_function=self.embedding,
            persist_directory=self.db_dir,
            collection_name=self.collection_name
        )
        
        

    async def load_file(self, file_path):
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
            doc = Document(page_content=content, metadata={"source": file_path})
            return [doc]
        else:
            log.err(f"Type de fichier non supporté : {ext}")

        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, loader.load)
        return data

    async def init_db(self):
        if not os.path.exists(self.db_dir):
            log.warn("Le répertoire de la base de données n'existe pas. Création du répertoire...")
            if not os.path.exists(self.files_dir):
                log.err("Le répertoire './db/' est introuvable. Veuillez vous assurer qu'il est présent dans le répertoire de travail.")
                return

            documents = []
            log.info(f"Chargement des fichiers PDF depuis le répertoire '{self.files_dir}' cela peut prendre un peu de temps...")
            for file_path in self.files:
                log.info(f"Fichier trouvé : {file_path}")
                data = await self.load_file(file_path)
                documents.extend(data)

            chunks = self.text_splitter.split_documents(documents)

            self.collection = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding,
                persist_directory=self.db_dir,
                collection_name=self.collection_name
            )
        else:
            log.info("Le répertoire de la base de données existe déjà.")
            log.info("Vérification des fichiers présents dans la collection...")
                
            self.collection = Chroma(
                embedding_function=self.embedding,
                persist_directory=self.db_dir,
                collection_name=self.collection_name
            )
            
            res = self.collection.get(include=["metadatas", "documents"])
            metadatas = [meta.get('source') for meta, doc in zip(res['metadatas'], res['documents'])]
            metadatas = list(set(metadatas))
            for file_path in self.files:
                if not file_path in metadatas:
                    log.warn(f"Le fichier '{file_path}' n'est pas présent dans la collection. Ajout en cours...")
                    full_path = os.path.join(self.files_dir, file_path)
                    data = await self.load_file(full_path)
                    chunks = self.text_splitter.split_documents(data)
                    for chunk in chunks:
                        chunk.metadata['source'] = file_path
                    
                    try:
                        self.collection.add_documents(documents=chunks)
                        log.success(f"Le fichier '{file_path}' a été ajouté avec succès à la collection.")
                    except Exception as e:
                        log.err(f"Erreur lors de l'ajout du fichier '{file_path}' à la collection : {e}")
                        return

        log.success("Collection chargée avec succès!")
