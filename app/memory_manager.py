from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
try:
    from app.config import MODEL_NAME
except ModuleNotFoundError:
    from config import MODEL_NAME

DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI assistant. The AI assistant is helpful, creative, clever, and very friendly.

Current conversation:
{history}

Human: {input}
AI: """

class MemoryManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            llm = ChatOpenAI(model=MODEL_NAME)
            cls._instance.memory = ConversationBufferMemory(return_messages=False)
            template = PromptTemplate(
                template=DEFAULT_TEMPLATE,
                input_variables=["history", "input"]
            )
            cls._instance.chain = ConversationChain(
                llm=llm,
                memory=cls._instance.memory,
                prompt=template,
                verbose=False
            )
        return cls._instance
    
    @classmethod
    def get_memory(cls):
        return cls().memory 

    def load_memory_variables(self, *args, **kwargs):
        return self.memory.load_memory_variables(*args, **kwargs)

    def save_context(self, *args, **kwargs):
        return self.memory.save_context(*args, **kwargs)
        
    def chat(self, user_input: str, file_path: str = None) -> str:
        """
        Process a chat message with optional file context.
        
        Args:
            user_input (str): The user's message
            file_path (str, optional): Path to the relevant file
            
        Returns:
            str: The assistant's response
        """
        try:
            if file_path:
                context = f"Using the document at {file_path}. "
                prompt = context + user_input
            else:
                prompt = user_input
                
            # Use predict instead of run for better handling
            response = self.chain.predict(input=prompt)
            
            if not response or response.strip() == "":
                return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                
            return response.strip()
            
        except Exception as e:
            return f"An error occurred: {str(e)}"