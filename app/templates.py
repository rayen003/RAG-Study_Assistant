from langchain_core.prompts import PromptTemplate

# Base conversation template that includes chat history
BASE_CONVERSATION_TEMPLATE = """
You are a helpful AI assistant. Using your knowledge and the conversation history, 
please provide a clear and informative response.

Previous conversation:
{chat_history}

Current question:
{question}

{additional_context}

Please provide a detailed but concise response.
"""

# Template configurations for different workflows
TEMPLATES = {
    "classifier": PromptTemplate(
        template="""You are a query classifier. Your task is to categorize the following query into exactly one of the available categories.

Query: {query}
Available categories: {categories}

STRICT Category Selection Rules:
1. Multimodal Processing: ALWAYS choose this for ANY queries about:
   - Images or pictures
   - Videos
   - Audio
   - Visual analysis
   - "Look at", "see", "analyze this image", "describe this picture"

2. Data Analysis: For queries about:
   - Data analysis
   - Statistics
   - Datasets
   - Numbers or trends

3. Content Generation: For requests to:
   - Write or create content
   - Generate text
   - Summarize articles
   - Create stories or posts

4. Advanced Processing: For:
   - Complex technical questions
   - Machine learning concepts
   - Advanced algorithms

5. Basic Processing: For:
   - Simple facts
   - Basic definitions
   - General knowledge

6. General Processing: ONLY for:
   - Greetings
   - Casual conversation
   - Non-specific queries

IMPORTANT: If the query mentions ANYTHING about images, pictures, or visual analysis, you MUST return "Multimodal Processing"

Return ONLY the category name, nothing else.
""",
        input_variables=["query", "categories"]
    ),
    
    "general": PromptTemplate(
        template=BASE_CONVERSATION_TEMPLATE.replace("{additional_context}", ""),
        input_variables=["chat_history", "question"]
    ),
    
    "rag": PromptTemplate(
        template=BASE_CONVERSATION_TEMPLATE.replace(
            "{additional_context}",
            "Context:\n{context}"
        ),
        input_variables=["chat_history", "question", "context"]
    ),
    
    "multimodal": PromptTemplate(
        template=BASE_CONVERSATION_TEMPLATE.replace(
            "{additional_context}",
            "Available context and images:\n{context}"
        ),
        input_variables=["chat_history", "question", "context"]
    )
} 