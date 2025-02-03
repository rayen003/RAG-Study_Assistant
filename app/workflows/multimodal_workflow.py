from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.string import StrOutputParser
try:
    from app.templates import TEMPLATES
    from app.config import MODEL_NAME
except ModuleNotFoundError:
    from ..templates import TEMPLATES
    from ..config import MODEL_NAME

def multimodal_workflow(memory, question, file_input=None):
    """Multimodal workflow for handling queries with file inputs.
    
    Args:
        memory: Memory manager instance
        question: User's question
        file_input: Optional file input containing file content and metadata
    """
    llm = ChatOpenAI(model=MODEL_NAME)

    try:
        # Process file content if available
        if file_input:
            if file_input.file_type == 'document':
                try:
                    # For text files, decode the content
                    file_content = file_input.content.decode('utf-8')
                    context = f"File content:\n{file_content}"
                except UnicodeDecodeError:
                    context = "Binary document content. Please ask specific questions about the document."
            else:  # image type
                context = "Image content. Please describe what you'd like to know about the image."
        else:
            context = "No file provided. Proceeding with general query."
        
        # Create and execute chain
        chain = (
            {
                "chat_history": lambda _: memory.load_memory_variables({}).get("chat_history", ""),
                "question": RunnablePassthrough(),
                "context": lambda _: context
            }
            | TEMPLATES["multimodal"]
            | llm
            | StrOutputParser()
        )
        
        response = chain.invoke(question)
        
        # Save to memory
        memory.save_context(
            {"question": question},
            {"answer": response}
        )
        
        return {
            "answer": response,
            "chat_history": memory.load_memory_variables({}).get("chat_history", "")
        }
            
    except Exception as e:
        error_msg = f"Error in multimodal processing: {str(e)}"
        return {
            "answer": error_msg,
            "chat_history": memory.load_memory_variables({}).get("chat_history", "")
        }