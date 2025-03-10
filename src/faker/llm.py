"""LLM provider interfaces for generating conversations."""

import os
import logging
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud.aiplatform import GenerationConfig

# Load environment variables
load_dotenv()


class LLMProvider(ABC):
    """Base abstract class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a completion for the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional parameters for generation
            
        Returns:
            The generated text
        """
        pass


class GeminiProvider(LLMProvider):
    """Provider for Google's Gemini via Vertex AI."""
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model: str = "gemini-1.5-pro",
    ):
        """Initialize the Gemini provider.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud location
            model: Gemini model to use
        """
        self.logger = logging.getLogger(__name__)
        
        # Load from environment variables if not provided
        self.project_id = project_id or os.getenv("PROJECT_ID")
        self.location = location or os.getenv("LOCATION", "us-central1")
        self.model = model
        
        if not self.project_id:
            raise ValueError("PROJECT_ID must be provided either directly or via environment variable")
        
        # Initialize Vertex AI
        aiplatform.init(project=self.project_id, location=self.location)
        
        # Get the model
        self.logger.info(f"Initializing Gemini model: {self.model}")
        self.client = aiplatform.GenerativeModel(self.model)
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using Gemini.
        
        Args:
            prompt: The prompt to send to Gemini
            **kwargs: Additional parameters for generation
            
        Returns:
            The generated text
        """
        self.logger.debug(f"Sending prompt to Gemini: {prompt[:100]}...")
        
        # Create generation config from kwargs
        generation_config = GenerationConfig(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
            top_k=kwargs.get("top_k", 40),
            max_output_tokens=kwargs.get("max_tokens", 1024),
        )
        
        # Generate response
        response = self.client.generate_content(
            prompt,
            generation_config=generation_config,
        )
        
        # Extract text from response
        result = response.text
        
        self.logger.debug(f"Received response from Gemini: {result[:100]}...")
        return result