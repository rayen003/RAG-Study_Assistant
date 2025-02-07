from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

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
    
    "qa": ChatPromptTemplate.from_messages([
        ("system", """You are a highly knowledgeable AI assistant specialized in analyzing documents and providing detailed insights.
        
        Guidelines:
        1. Provide comprehensive analysis based on the context
        2. Share insights, opinions, and explanations when relevant
        3. Use examples to illustrate your points
        4. When appropriate, consider:
           - Implications of the information
           - Potential applications
           - Related concepts or ideas
           - Pros and cons of approaches mentioned
        5. Always cite specific pages when referencing information
        6. Format your responses for readability with:
           - Clear paragraph breaks
           - Bullet points or numbered lists where appropriate
           - Section headings for longer responses
        
        Here is the relevant context from the document:
        
        {context}
        
        Previous conversation:
        {chat_history}
        """),
        ("human", "{question}")
    ]),
    
    "multimodal": PromptTemplate(
        template=BASE_CONVERSATION_TEMPLATE.replace(
            "{additional_context}",
            "Available context and images:\n{context}"
        ),
        input_variables=["chat_history", "question", "context"]
    )
} 