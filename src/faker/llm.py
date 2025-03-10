"""LLM provider interfaces for generating conversations."""

import os
import time
import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud.aiplatform import GenerationConfig
from google.oauth2 import service_account

# Load environment variables
load_dotenv()


class LLMProvider(ABC):
    """Base abstract class for LLM providers."""
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a completion for the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt/instructions
            **kwargs: Additional parameters for generation
            
        Returns:
            A tuple containing (generated_text, response_metadata)
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            A dictionary with model information
        """
        pass


class GeminiProvider(LLMProvider):
    """Provider for Google's Gemini via Vertex AI."""
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model: str = "gemini-1.5-pro",
        credentials_path: Optional[str] = None,
        default_temperature: float = 0.7,
        default_top_p: float = 0.95,
        default_top_k: int = 40,
        default_max_tokens: int = 1024,
    ):
        """Initialize the Gemini provider.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud location
            model: Gemini model to use
            credentials_path: Path to service account credentials
            default_temperature: Default temperature for generation
            default_top_p: Default top_p for generation
            default_top_k: Default top_k for generation
            default_max_tokens: Default maximum tokens for generation
        """
        self.logger = logging.getLogger(__name__)
        
        # Load from environment variables if not provided
        self.project_id = project_id or os.getenv("PROJECT_ID")
        self.location = location or os.getenv("LOCATION", "us-central1")
        self.model = model
        self.credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Default generation parameters
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.default_top_k = default_top_k
        self.default_max_tokens = default_max_tokens
        
        if not self.project_id:
            raise ValueError("PROJECT_ID must be provided either directly or via environment variable")
        
        # Initialize Vertex AI
        self._initialize_client()
        
    def _initialize_client(self) -> None:
        """Initialize the Vertex AI client."""
        self.logger.info(f"Initializing Gemini model: {self.model}")
        
        # If credentials path is provided, use it
        if self.credentials_path:
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                aiplatform.init(
                    project=self.project_id, 
                    location=self.location,
                    credentials=credentials
                )
            except Exception as e:
                self.logger.warning(f"Failed to load credentials from {self.credentials_path}: {e}")
                self.logger.warning("Falling back to default credentials")
                aiplatform.init(project=self.project_id, location=self.location)
        else:
            aiplatform.init(project=self.project_id, location=self.location)
        
        # Get the model
        self.client = aiplatform.GenerativeModel(self.model)
        
    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a response using Gemini.
        
        Args:
            prompt: The prompt to send to Gemini
            system_prompt: Optional system prompt/instructions
            **kwargs: Additional parameters for generation
            
        Returns:
            A tuple containing (generated_text, response_metadata)
        """
        self.logger.debug(f"Sending prompt to Gemini: {prompt[:100]}...")
        
        # Track timing
        start_time = time.time()
        
        # Create generation config from kwargs
        generation_config = GenerationConfig(
            temperature=kwargs.get("temperature", self.default_temperature),
            top_p=kwargs.get("top_p", self.default_top_p),
            top_k=kwargs.get("top_k", self.default_top_k),
            max_output_tokens=kwargs.get("max_tokens", self.default_max_tokens),
            candidate_count=kwargs.get("candidate_count", 1),
        )
        
        # Prepare the chat history if system prompt is provided
        if system_prompt:
            chat = self.client.start_chat(
                system_instructions=system_prompt,
            )
            response = chat.send_message(
                prompt,
                generation_config=generation_config,
            )
        else:
            # Generate response without system prompt
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config,
            )
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Extract text from response
        result = response.text
        
        # Collect metadata from the response
        metadata = {
            "model": self.model,
            "generation_time": generation_time,
            "generation_config": {
                "temperature": generation_config.temperature,
                "top_p": generation_config.top_p,
                "top_k": generation_config.top_k,
                "max_tokens": generation_config.max_output_tokens,
            },
            "provider": "gemini"
        }
        
        # Add additional response metadata if available
        if hasattr(response, "prompt_feedback"):
            metadata["prompt_feedback"] = response.prompt_feedback.to_dict()
            
        self.logger.debug(f"Received response from Gemini: {result[:100]}...")
        return result, metadata
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Gemini model.
        
        Returns:
            A dictionary with model information
        """
        return {
            "name": self.model,
            "provider": "gemini",
            "project_id": self.project_id,
            "location": self.location
        }


class MockProvider(LLMProvider):
    """Mock provider for testing without API calls."""
    
    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = '{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}',
    ):
        """Initialize the mock provider.
        
        Args:
            responses: Dictionary mapping prompt substrings to responses
            default_response: Default response if no match is found
        """
        self.logger = logging.getLogger(__name__)
        self.responses = responses or {}
        self.default_response = default_response
        
    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a mock response.
        
        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt/instructions
            **kwargs: Additional parameters for generation
            
        Returns:
            A tuple containing (generated_text, response_metadata)
        """
        self.logger.debug(f"Mock provider received prompt: {prompt[:100]}...")
        
        # Find a matching response
        for key, response in self.responses.items():
            if key in prompt:
                return response, {"provider": "mock", "matched_key": key}
                
        # Return default response
        return self.default_response, {"provider": "mock", "matched_key": None}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the mock model.
        
        Returns:
            A dictionary with model information
        """
        return {
            "name": "mock-provider",
            "provider": "mock"
        }