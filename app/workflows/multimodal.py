from .base import BaseWorkflow
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

class MultimodalWorkflow(BaseWorkflow):
    def __init__(self, vector_store):
        self.retrieval_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
    
    def handle_query(self, query: str) -> dict:
        result = self.retrieval_chain({"query": query})
        return {
            "answer": result['result'],
            "sources": result.get('source_documents', [])
        }
