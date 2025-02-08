from .base import BaseWorkflow
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from app.config import OPENAI_API_KEY, MODEL_NAME, MODEL_TEMPERATURE

class GeneralWorkflow(BaseWorkflow):
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=MODEL_NAME,
            temperature=MODEL_TEMPERATURE
        )
        self.system_message = SystemMessage(content="You are a helpful AI assistant.")
    
    def __call__(self, inputs: dict) -> dict:
        try:
            if "input" in inputs:  # Chat mode
                messages = [
                    self.system_message,
                    HumanMessage(content=inputs["input"])
                ]
                response = self.llm.invoke(messages)
                return {
                    "response": response.content
                }
            else:  # QA mode
                return self.handle_query(inputs["question"])
        except Exception as e:
            print(f"Error in GeneralWorkflow.__call__: {str(e)}")
            raise
    
    def handle_query(self, query: str) -> dict:
        try:
            messages = [
                self.system_message,
                HumanMessage(content=query)
            ]
            response = self.llm.invoke(messages)
            return {
                "answer": response.content,
                "source_documents": []
            }
        except Exception as e:
            print(f"Error in GeneralWorkflow.handle_query: {str(e)}")
            raise
