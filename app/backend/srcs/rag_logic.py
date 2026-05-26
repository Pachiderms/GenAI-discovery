from langchain_classic.chains import RetrievalQA
from langchain_ollama import OllamaLLM

from file_handler import FileHandler


class RAGLogic:
    def __init__(self, file_handler: FileHandler):
        self.file_handler = file_handler
        self.retriever = self.file_handler.collection.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.25, "score_threshold": 0.2}
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=OllamaLLM(model="mistral"),
            chain_type="stuff",
            retriever=self.retriever
        )

    async def ask(self, question: str) -> str:
        answer = self.qa_chain.invoke(question)
        return answer