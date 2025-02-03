from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
try:
    from app.config import MODEL_NAME
except ModuleNotFoundError:
    from config import MODEL_NAME

def create_memory():
    """
    Create a conversation summary memory.
    """
    return ConversationSummaryMemory(
        llm=ChatOpenAI(model=MODEL_NAME),
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )

class MemoryManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.memory = create_memory()
        return cls._instance
    
    @classmethod
    def get_memory(cls):
        return cls().memory 

    def load_memory_variables(self, *args, **kwargs):
        return self.memory.load_memory_variables(*args, **kwargs)

    def save_context(self, *args, **kwargs):
        return self.memory.save_context(*args, **kwargs)