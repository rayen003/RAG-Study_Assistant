from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
try:
    from app.templates import TEMPLATES
    from app.config import MODEL_NAME
except ModuleNotFoundError:
    from ..templates import TEMPLATES
    from ..config import MODEL_NAME

def general_workflow(memory, question, file_input=None):
    """General workflow for text-based queries.
    
    Args:
        memory: Memory manager instance
        question: User's question
        file_input: Optional file input (ignored in general workflow)
    """
    llm = ChatOpenAI(model=MODEL_NAME)

    chain = (
        {
            "chat_history": lambda _: memory.load_memory_variables({}).get("chat_history", ""),
            "question": RunnablePassthrough()
        }
        | TEMPLATES["general"]
        | llm
        | StrOutputParser()
    )

    # Execute the chain and get the response
    response = chain.invoke(question)

    # Save to memory
    memory.save_context(
        {"question": question},
        {"answer": response}
    )

    # Format the final output
    return {
        "answer": response,
        "chat_history": memory.load_memory_variables({}).get("chat_history", "")
    }

# Example test usage
if __name__ == "__main__":
    # Create test memory
    test_memory = ConversationSummaryMemory(
        llm=ChatOpenAI(model=MODEL_NAME),
        memory_key="chat_history",
        input_key="question", 
        output_key="answer",
        return_messages=True
    )
    
    # Test questions
    test_questions = [
        "What is machine learning?",
        "Can you explain neural networks?",
        "How does backpropagation work?"
    ]
    
    # Run workflow with test questions
    for question in test_questions:
        print(f"\nQuestion: {question}")
        response = general_workflow(test_memory, question)
        print(f"Answer: {response['answer']}")
