from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from colored_print import log

from file_handler import FileHandler
from prompt_templates import system_prompt, nlp_prompt
import json

class RAGLogic:
    def __init__(self, file_handler: FileHandler):
        self.file_handler = file_handler
        self.system_prompt = PromptTemplate.from_template(system_prompt)
        self.retriever = self.file_handler.collection.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.25, "score_threshold": 0.2},
            temperature=0,
        )
        
        self.llm = OllamaLLM(model="mistral:7b", temperature=0.0, max_tokens=2000)
                
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever
        )
        
        self.history = []
        
        self.warmup()
        
    def warmup(self):
        log.info("Model warmup")
        dummy_query = "test"
        
        docs = self.retriever.invoke(dummy_query)
        _ = self.llm.invoke("Bonjour")
        log.success("✅ Model prêt")

        
    def ner_score(self, doc, entities):
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


    def recognition(self, query):
        doc = self.file_handler.nlp(query)
        
        return [
            {
                'text': ent.text,
                'label': ent.label_
            }
            for ent in doc.ents
        ]


    def preproc_query(self, query):
        
        augmented_query = json.loads(self.llm.invoke(nlp_prompt.format(question=query)))
        log.info(f"{augmented_query=}")
        
        
        new_query = augmented_query['answer']
        docs = self.retriever.invoke(new_query)
        entities = self.recognition(new_query)
        
        docs = sorted(
            docs,
            key=lambda d: self.ner_score(d, entities),
            reverse=True
        )
        
        docs = docs[:5]
        
        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )
        
        final_query = system_prompt.format(chat_history=self.history[-5:], context=context, question=new_query)
        
        answer = self.llm.invoke(final_query)
        
        self.history.append({
            'question': new_query,
            'answer': answer
        })
        
        log.info(f"{context=}\n {self.history=}")
        
        return {
            'context': context,
            'answer': answer,
        }

    async def ask(self, question: str) -> str:
        log.warn(f"Question reçue : {question}")
        try:
            answer = self.preproc_query(question)
            log.info(f"Réponse générée : {answer}")
            return answer.get("answer", "Désolé, je n'ai pas pu trouver une réponse à votre question.")
        except Exception as e:
            log.err(f"Erreur lors de la génération de la réponse : {e}")
            return "Désolé, une erreur est survenue lors de la génération de la réponse."
