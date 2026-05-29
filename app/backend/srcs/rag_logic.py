from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from colored_print import log

from file_handler import FileHandler
from prompt_templates import system_prompt
import json

class RAGLogic:
    def __init__(self, file_handler: FileHandler):
        self.file_handler = file_handler
        self.system_prompt = PromptTemplate.from_template(system_prompt)
        self.retriever = self.file_handler.collection.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.25, "score_threshold": 0.0}
        )
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", answer_key="answer", return_messages=True)
        llm = OllamaLLM(model="mistral:7b", temperature=0.0, max_tokens=1000)
                
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.system_prompt},
        )

    async def ask(self, question: str) -> str:
        log.warn(f"Question reçue : {question}")
        try:
            answer = self.qa_chain.invoke({"question": question})
            log.success(f"Réponse générée : {answer.get('answer')}")
            return answer.get("answer", "Désolé, je n'ai pas pu trouver une réponse à votre question.")
        except Exception as e:
            log.err(f"Erreur lors de la génération de la réponse : {e}")
            return "Désolé, une erreur est survenue lors de la génération de la réponse."
