from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import logging
from typing import Dict, List, Optional
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from app.config import MODEL_NAME
except ModuleNotFoundError:
    from config import MODEL_NAME

class ChatManager:
    def __init__(self, model_name=MODEL_NAME):
        logger.info("Initializing ChatManager...")
        self.model_name = model_name
        self.messages = []
        
        # Create memory with the right input/output keys
        memory = ConversationBufferMemory(
            input_key="input",
            memory_key="history",
            return_messages=False
        )
        
        # Create chat prompt template
        prompt = PromptTemplate(
            input_variables=["history", "input", "context"],
            template="""You are a helpful AI assistant analyzing documents. 
            When provided with context, use it to give accurate and relevant answers.
            If no context is provided or the context doesn't contain relevant information, 
            let the user know and answer based on your general knowledge.

            Context: {context}

            Chat History:
            {history}

            Current Question: {input}

            Assistant: Let me help you with that."""
        )

        # Initialize conversation chain with memory
        self.conversation = LLMChain(
            llm=ChatOpenAI(model=model_name),
            prompt=prompt,
            memory=memory,
            verbose=True
        )
        
        logger.info("ChatManager initialization complete.")

    def get_response(self, query: str, context: str = "") -> str:
        """Get response from the model"""
        try:
            # Add user message to history
            self.messages.append({"role": "user", "content": query})
            
            # Format history for the prompt
            history = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}" 
                for msg in self.messages[-5:]  # Only use last 5 messages
            ])
            
            # Get response from conversation chain
            logger.info("Getting response with context length: %d", len(context))
            response = self.conversation.predict(
                input=query,
                context=context or "No specific document context available.",
                history=history
            )
            
            # Add response to history
            self.messages.append({"role": "assistant", "content": response.strip()})
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            raise e

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        logger.info("Adding message to conversation history: %s: %s", role.capitalize(), content)
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> List[str]:
        """Get all messages in the conversation history."""
        logger.info("Retrieving conversation history")
        return [f"{message['role'].capitalize()}: {message['content']}" for message in self.messages]

    def clear_messages(self):
        """Clear all messages from the conversation history."""
        logger.info("Clearing conversation history")
        self.messages = []

    def chat(self, user_input: str, file_path: str = None) -> str:
        """Process a chat message with optional file context."""
        try:
            # Get document context if file_path is provided
            if file_path:
                logger.info("Loading vector store from: %s", file_path)
                st.info("Loading document context...")
                
                from app.rag import load_vector_store
                vector_store = load_vector_store(file_path)
                
                if not vector_store:
                    logger.error("Failed to load vector store")
                    return "Sorry, I couldn't load the document. Please try uploading it again."
                
                logger.info("Searching for relevant content...")
                st.info("Finding relevant document sections...")
                
                # Search for relevant content
                docs = vector_store.similarity_search(user_input, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                logger.info("Found %d relevant document sections", len(docs))
            else:
                logger.info("No file path provided, proceeding without document context")
                context = ""
            
            # Add user message and get response
            logger.info("Processing user message...")
            st.info("Generating response...")
            
            self.add_message("user", user_input)
            response = self.get_response(user_input, context)
            
            if not response or response.strip() == "":
                logger.warning("Empty response received")
                response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
            # Add assistant response
            self.add_message("assistant", response)
            logger.info("Chat processing complete")
            return response
            
        except Exception as e:
            logger.error("Error in chat: %s", str(e), exc_info=True)
            return f"An error occurred: {str(e)}"

    def reset_conversation(self):
        logger.info("Resetting conversation")
        self.messages = []

    def save_conversation(self, filename):
        logger.info("Saving conversation to file: %s", filename)
        with open(filename, 'w') as f:
            for message in self.get_messages():
                f.write(message + '\n')

    def load_conversation(self, filename):
        logger.info("Loading conversation from file: %s", filename)
        try:
            with open(filename, 'r') as f:
                messages = f.readlines()
                for message in messages:
                    role, content = message.strip().split(": ", 1)
                    self.messages.append({"role": role.lower(), "content": content})
        except FileNotFoundError:
            logger.info("Conversation file not found.")

    def get_conversation_history(self):
        logger.info("Retrieving conversation history")
        return self.get_messages()

    def handle_message(self, message: str) -> str:
        """
        Handle a message from the user.
        
        Args:
            message (str): The user's message
        
        Returns:
            str: The assistant's response
        """
        logger.info("Handling user message: %s", message)
        self.add_message("user", message)
        response = self.get_response(message)
        self.add_message("assistant", response)
        return response

    def get_history(self) -> List[str]:
        """
        Get the conversation history.
        
        Returns:
            List[str]: The conversation history
        """
        logger.info("Retrieving conversation history")
        return self.get_conversation_history()
