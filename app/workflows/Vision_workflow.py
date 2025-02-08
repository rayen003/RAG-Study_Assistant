"""
Vision workflow for handling image-based queries
"""
import logging
import base64
from typing import Dict, Any, List, Union
from pathlib import Path
import torch
from PIL import Image
from openai import OpenAI
import numpy as np
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain

from app.workflows.base import BaseWorkflow
from app.models.image_embeddings import get_image_embeddings
from app.core.memory_manager import ConversationMemory

# Configure logging
logger = logging.getLogger(__name__)

class VisionWorkflow(BaseWorkflow):
    """Workflow for handling image-based queries"""
    
    def __init__(self, memory: ConversationMemory):
        logger.info("Initializing VisionWorkflow")
        super().__init__(memory)
        self.image_embeddings = get_image_embeddings()
        self.openai_client = OpenAI()
        self.vision_template = self._create_vision_template()
        self.vision_chain = self._create_vision_llm_chain()
        logger.info("VisionWorkflow initialized successfully")
    
    def _create_vision_template(self) -> ChatPromptTemplate:
        """Create vision-specific template"""
        system_template = """You are an expert image analysis AI assistant. Your task is to provide detailed, 
accurate, and contextually relevant analysis of images while maintaining conversation context.

Previous conversation:
{chat_history}

Guidelines for analysis:
1. Consider the specific query and provide targeted analysis
2. Describe key visual elements:
   - Objects and subjects
   - Actions and activities
   - Setting and environment
   - Colors and visual composition
   - Text or symbols if present
3. Maintain context from previous interactions
4. Be precise but natural in your descriptions
5. If uncertainty exists, acknowledge it

Remember to:
- Stay focused on the query
- Use clear, descriptive language
- Reference previous context when relevant
- Be specific about visual details"""

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{query}\n\nImage Analysis: {image_description}")
        
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
        
        return chat_prompt
    
    def _create_vision_llm_chain(self):
        """Create vision-specific LLM chain"""
        return LLMChain(
            llm=self.llm,
            prompt=self.vision_template
        )
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load an image from a file path"""
        logger.info(f"Loading image from: {image_path}")
        try:
            image = Image.open(image_path)
            logger.info(f"Image loaded successfully. Size: {image.size}, Mode: {image.mode}")
            return image
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise
    
    def process_image(self, image_path: str) -> torch.Tensor:
        """Process image and generate embeddings"""
        logger.info(f"Processing image for embeddings: {image_path}")
        image = self.load_image(image_path)
        embeddings = self.image_embeddings.embed_images(image)
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def cosine_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """Calculate cosine similarity between two embeddings"""
        embedding1 = embedding1 / embedding1.norm(dim=-1, keepdim=True)
        embedding2 = embedding2 / embedding2.norm(dim=-1, keepdim=True)
        return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=1).item()
    
    def _generate_fallback_description(self, image_path: str, query: str) -> str:
        """Generate a fallback description using CLIP embeddings and basic analysis"""
        image = self.load_image(image_path)
        
        # Get dominant colors
        colors = image.getcolors(image.size[0] * image.size[1])
        if colors:
            dominant_color = max(colors, key=lambda x: x[0])[1]
            color_info = f"Dominant colors: RGB{dominant_color}"
        else:
            color_info = "No dominant color found"
        
        # Get embedding statistics
        embeddings = self.process_image(image_path)
        embedding_stats = f"Embedding stats: Mean={embeddings.mean().item():.2f}, Std={embeddings.std().item():.2f}"
        
        return f"{color_info}\n{embedding_stats}"
    
    def handle_query(self, query: str, image_path: str = None, **kwargs) -> Dict[str, Any]:
        """Handle an image-based query"""
        logger.info(f"Handling vision query: {query}")
        
        if not image_path:
            raise ValueError("Image path is required for vision workflow")
        
        try:
            logger.info("Starting image analysis")
            
            # Process image for embeddings
            embeddings = self.process_image(image_path)
            
            # Prepare image for OpenAI
            base64_image = self.encode_image(image_path)
            
            # Prepare messages for OpenAI
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            logger.info("Sending request to OpenAI Vision model")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=300
            )
            logger.info("Received response from OpenAI")
            
            # Get the response text
            answer = response.choices[0].message.content
            
            # Update conversation memory
            if self.memory:
                self.memory.add_interaction(query, answer)
            
            # Prepare context for LLM chain
            context = {
                "chat_history": self.memory.get_chat_history() if self.memory else "",
                "image_description": answer,
                "query": query
            }
            
            # Generate enhanced response using LLM chain
            enhanced_response = self.vision_chain.invoke(context)["text"]
            
            return {
                "answer_generator": enhanced_response,
                "sources": [{"type": "image", "path": image_path}]
            }
            
        except Exception as e:
            logger.error(f"Error in vision workflow: {str(e)}")
            logger.info("Using fallback description generation")
            
            # Generate fallback description
            fallback_desc = self._generate_fallback_description(image_path, query)
            
            # Use base chain for fallback
            context = {
                "chat_history": self.memory.get_chat_history() if self.memory else "",
                "query": query,
                "additional_context": f"Fallback image analysis:\n{fallback_desc}"
            }
            
            fallback_response = self.chain.invoke(context)["text"]
            
            return {
                "answer_generator": fallback_response,
                "sources": [{"type": "image", "path": image_path}]
            }
