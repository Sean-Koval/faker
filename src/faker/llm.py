"""LLM provider interfaces for generating conversations.

This module provides a flexible interface for interacting with different LLM providers.
New providers can be added by implementing the LLMProvider abstract base class.
"""

import dataclasses
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from typing import (Any, ClassVar, Dict, List, Optional, Protocol, Tuple, Type,
                    Union)
import threading
from datetime import datetime, timedelta

# Vertex AI imports
import vertexai
# Environment variables
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.oauth2 import service_account
from vertexai.generative_models import GenerationConfig, GenerativeModel, Part

# Import performance timer
from src.faker.logging_service import PerformanceTimer, timer, async_timer

# Import errors module for exception handling
try:
    from google.cloud.aiplatform import errors as aiplatform_errors
except ImportError:
    # Creating a mock module for errors if it can't be imported
    class MockErrorsModule:
        class ResourceExhausted(Exception): pass
        class ServiceUnavailable(Exception): pass
    aiplatform_errors = MockErrorsModule

# Load environment variables
load_dotenv()

# Try to import additional packages, but provide graceful fallbacks
try:
    import backoff
    from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                          wait_exponential)

    HAS_RETRY_LIBS = True
except ImportError:
    HAS_RETRY_LIBS = False

try:
    from vertexai.preview.generative_models import Image

    HAS_IMAGE_SUPPORT = True
except ImportError:
    HAS_IMAGE_SUPPORT = False

# Try to import additional packages for concurrency
try:
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor
    HAS_CONCURRENT_FUTURES = True
except ImportError:
    HAS_CONCURRENT_FUTURES = False


class GenerationParameters:
    """Standard parameters for generation across different providers."""

    def __init__(
        self,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        max_tokens: int = 1024,
        stop_sequences: Optional[List[str]] = None,
        candidate_count: int = 1,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: Optional[int] = None,
        **additional_params,
    ):
        """Initialize generation parameters.

        Args:
            temperature: Controls randomness. Higher values (e.g., 0.8) make output more random,
                         lower values (e.g., 0.2) make it more focused and deterministic.
            top_p: Nucleus sampling. Consider the most likely tokens whose probability sum is <= top_p.
            top_k: Consider only the top k most likely tokens.
            max_tokens: Maximum number of tokens to generate.
            stop_sequences: List of sequences that, when generated, will stop generation.
            candidate_count: Number of candidate responses to generate.
            presence_penalty: Penalizes tokens that have already appeared in the text.
            frequency_penalty: Penalizes tokens that appear frequently in the text.
            repetition_penalty: Penalizes repetition of token sequences.
            random_seed: Optional seed for deterministic generation.
            additional_params: Any additional parameters specific to certain providers.
        """
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences or []
        self.candidate_count = candidate_count
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty
        self.random_seed = random_seed
        self.additional_params = additional_params

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "stop_sequences": self.stop_sequences,
            "candidate_count": self.candidate_count,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
        }

        if self.random_seed is not None:
            result["random_seed"] = self.random_seed

        # Add any additional parameters
        result.update(self.additional_params)

        return result


class GenerationResult:
    """Result of a generation request."""

    def __init__(
        self,
        text: str,
        model: str,
        provider: str,
        generation_time: float,
        finish_reason: Optional[str] = None,
        generation_params: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, int]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a generation result.

        Args:
            text: The generated text.
            model: The model used for generation.
            provider: The provider name.
            generation_time: Time taken for generation in seconds.
            finish_reason: Reason for finishing generation (if provided).
            generation_params: Parameters used for generation.
            usage: Token usage statistics (if available).
            additional_info: Any additional info from the provider.
        """
        self.text = text
        self.model = model
        self.provider = provider
        self.generation_time = generation_time
        self.finish_reason = finish_reason
        self.generation_params = generation_params or {}
        self.usage = usage or {}
        self.additional_info = additional_info or {}
        self.timestamp = time.time()
        self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "generation_time": self.generation_time,
            "finish_reason": self.finish_reason,
            "generation_params": self.generation_params,
            "usage": self.usage,
            "additional_info": self.additional_info,
            "timestamp": self.timestamp,
        }


class ProviderType(str, Enum):
    """Supported provider types."""

    VERTEX_AI = "vertex_ai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGING_FACE = "hugging_face"
    MOCK = "mock"
    CUSTOM = "custom"


class RateLimiter:
    """Rate limiter for API calls to prevent exceeding quotas."""
    
    def __init__(self, calls_per_minute: int = 60, max_parallel: int = 10):
        """Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum number of calls allowed per minute
            max_parallel: Maximum number of parallel requests allowed
        """
        self.calls_per_minute = calls_per_minute
        self.max_parallel = max_parallel
        self.call_timestamps = []
        self.lock = threading.RLock()
        self.semaphore = threading.Semaphore(max_parallel)
        
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        with self.lock:
            now = datetime.now()
            
            # Remove timestamps older than 1 minute
            self.call_timestamps = [
                ts for ts in self.call_timestamps 
                if now - ts < timedelta(minutes=1)
            ]
            
            # Check if we're at the rate limit
            if len(self.call_timestamps) >= self.calls_per_minute:
                # Calculate required wait time 
                oldest = min(self.call_timestamps)
                wait_time = (oldest + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    time.sleep(wait_time)
                    
                    # Clear outdated timestamps after waiting
                    now = datetime.now()
                    self.call_timestamps = [
                        ts for ts in self.call_timestamps 
                        if now - ts < timedelta(minutes=1)
                    ]
            
            # Record this call
            self.call_timestamps.append(now)
    
    def acquire(self) -> None:
        """Acquire permission to make an API call."""
        self.semaphore.acquire()
        self.wait_if_needed()
    
    def release(self) -> None:
        """Release the semaphore after making an API call."""
        self.semaphore.release()
        
    def __enter__(self):
        self.acquire()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class ResponseCache:
    """Cache for LLM responses to avoid duplicate calls."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize response cache.
        
        Args:
            max_size: Maximum number of responses to cache
        """
        self.cache = {}  # Dict mapping prompt -> (response, metadata)
        self.max_size = max_size
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Get a cached response.
        
        Args:
            key: Cache key (typically a hash of prompt and params)
            
        Returns:
            Tuple of (response, metadata) if cached, None otherwise
        """
        with self.lock:
            return self.cache.get(key)
    
    def put(self, key: str, value: Tuple[str, Dict[str, Any]]) -> None:
        """Add a response to the cache.
        
        Args:
            key: Cache key (typically a hash of prompt and params)
            value: Tuple of (response, metadata) to cache
        """
        with self.lock:
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                # Remove a random 10% of entries when cache is full
                keys_to_remove = list(self.cache.keys())[:self.max_size // 10]
                for k in keys_to_remove:
                    self.cache.pop(k, None)
            
            self.cache[key] = value


class LLMProvider(ABC):
    """Base abstract class for LLM providers."""

    provider_type: ClassVar[ProviderType]
    
    def __init__(self):
        """Initialize the provider with performance optimizations."""
        # Default rate limiter (can be overridden by subclasses)
        self.rate_limiter = RateLimiter()
        
        # Response cache
        self.response_cache = ResponseCache()
        
        # Flag to enable/disable caching
        self.use_cache = True

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        parameters: Optional[GenerationParameters] = None,
        retry_on_error: bool = True,
        use_cache: bool = True,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a completion for the given prompt.

        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt/instructions
            parameters: Generation parameters to use
            retry_on_error: Whether to retry on transient errors
            use_cache: Whether to use response caching
            **kwargs: Additional parameters for generation

        Returns:
            A tuple containing (generated_text, response_metadata)
        """
        pass
    
    def generate_batch(
        self, 
        prompts: List[str],
        system_prompt: Optional[str] = None,
        parameters: Optional[GenerationParameters] = None,
        max_parallel: int = 5,
        **kwargs
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate multiple completions in parallel.
        
        Args:
            prompts: List of prompts to send to the LLM
            system_prompt: Optional system prompt/instructions for all prompts
            parameters: Generation parameters to use
            max_parallel: Maximum number of parallel requests
            **kwargs: Additional parameters for generation
            
        Returns:
            List of tuples containing (generated_text, response_metadata)
        """
        if not HAS_CONCURRENT_FUTURES:
            # Fall back to sequential processing
            return [
                self.generate(prompt, system_prompt, parameters, **kwargs)
                for prompt in prompts
            ]
        
        results = [None] * len(prompts)
        
        # Use a smaller thread pool than max_parallel
        with ThreadPoolExecutor(max_workers=min(max_parallel, len(prompts))) as executor:
            # Submit all jobs
            future_to_idx = {
                executor.submit(
                    self.generate, 
                    prompt, 
                    system_prompt, 
                    parameters,
                    **kwargs
                ): i 
                for i, prompt in enumerate(prompts)
            }
            
            # Collect results, maintaining original order
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # Return error information in metadata
                    error_metadata = {
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    results[idx] = ("", error_metadata)
        
        return results

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.

        Returns:
            A dictionary with model information
        """
        pass

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self.provider_type.value
        
    def _get_cache_key(
        self, 
        prompt: str, 
        system_prompt: Optional[str], 
        parameters: Optional[Dict[str, Any]]
    ) -> str:
        """Generate a cache key for the given input parameters.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt/instructions
            parameters: Generation parameters
            
        Returns:
            A string key for caching
        """
        # Create a stable representation of the parameters
        key_parts = [
            prompt,
            system_prompt or "",
            json.dumps(parameters or {}, sort_keys=True)
        ]
        
        # Simple hash function, could replace with more sophisticated hash
        return str(hash("".join(key_parts)))


# Registry of provider types to provider classes
PROVIDER_REGISTRY: Dict[ProviderType, Type[LLMProvider]] = {}


def register_provider(provider_cls: Type[LLMProvider]) -> Type[LLMProvider]:
    """Register a provider class in the provider registry.

    Args:
        provider_cls: The provider class to register

    Returns:
        The registered provider class
    """
    if not hasattr(provider_cls, "provider_type"):
        raise ValueError(
            f"Provider class {provider_cls.__name__} must have a provider_type attribute"
        )

    PROVIDER_REGISTRY[provider_cls.provider_type] = provider_cls
    return provider_cls


@register_provider
class VertexAIProvider(LLMProvider):
    """Provider for Google's Vertex AI models."""

    provider_type = ProviderType.VERTEX_AI

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model: str = "gemini-1.5-pro",
        credentials_path: Optional[str] = None,
        default_parameters: Optional[GenerationParameters] = None,
        timeout: float = 300.0,
        max_retries: int = 3,
        vertex_endpoint: Optional[str] = None,
        calls_per_minute: int = 60,
        max_parallel_calls: int = 10,
        use_cache: bool = True,
        cache_size: int = 1000,
    ):
        """Initialize the Vertex AI provider.

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud location
            model: Model to use (e.g., gemini-1.5-pro)
            credentials_path: Path to service account credentials
            default_parameters: Default generation parameters to use
            timeout: Timeout for requests in seconds
            max_retries: Maximum number of retries for failed requests
            vertex_endpoint: Optional custom Vertex AI endpoint
            calls_per_minute: Maximum API calls per minute (rate limit)
            max_parallel_calls: Maximum parallel API calls allowed
            use_cache: Whether to enable response caching
            cache_size: Maximum size of response cache
        """
        # Initialize base class
        super().__init__()
        
        self.logger = logging.getLogger(__name__)

        # Load from environment variables if not provided
        self.project_id = project_id or os.getenv("PROJECT_ID")
        self.location = location or os.getenv("LOCATION", "us-central1")
        self.model = model
        self.credentials_path = credentials_path or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        self.vertex_endpoint = vertex_endpoint

        # Generation settings
        self.default_parameters = default_parameters or GenerationParameters()
        self.timeout = timeout
        self.max_retries = max_retries
        self._model_info: Optional[Dict[str, Any]] = None
        
        # Performance optimization settings
        self.rate_limiter = RateLimiter(
            calls_per_minute=calls_per_minute, 
            max_parallel=max_parallel_calls
        )
        self.response_cache = ResponseCache(max_size=cache_size)
        self.use_cache = use_cache

        if not self.project_id:
            raise ValueError(
                "PROJECT_ID must be provided either directly or via environment variable"
            )

        # Initialize Vertex AI
        self._initialize_client()

        # Cache for model info
        self._model_info = None

    def _initialize_client(self) -> None:
        """Initialize the Vertex AI client."""
        self.logger.info(f"Initializing Vertex AI model: {self.model}")

        # Load credentials if available
        credentials = None
        if self.credentials_path:
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self.logger.info(f"Loaded credentials from {self.credentials_path}")
            except Exception as e:
                self.logger.warning(
                    f"Failed to load credentials from {self.credentials_path}: {e}"
                )
                self.logger.warning("Falling back to default credentials")

        # Initialize Vertex AI
        try:
            # Initialize vertexai package
            vertexai.init(
                project=self.project_id, location=self.location, credentials=credentials
            )

            # Initialize aiplatform package
            aiplatform.init(
                project=self.project_id, location=self.location, credentials=credentials
            )

            # Create the model client
            self.client = GenerativeModel(self.model)
            self.logger.info(f"Successfully initialized Vertex AI model: {self.model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Vertex AI: {e}")
            raise

    def _create_generation_config(
        self, parameters: Optional[GenerationParameters] = None
    ) -> GenerationConfig:
        """Create a Vertex AI generation config from parameters.

        Args:
            parameters: Generation parameters to use (falls back to defaults if None)

        Returns:
            Vertex AI GenerationConfig
        """
        params = parameters or self.default_parameters

        # Create a cleaner set of parameters - only include parameters that are actually
        # supported by the current version of the Vertex AI SDK
        config_kwargs = {}
        
        # Try adding each parameter with error handling
        try:
            config = GenerationConfig()
            
            # Test if each attribute exists on the GenerationConfig class
            # Only add parameters that are supported
            for key, value in {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "max_output_tokens": params.max_tokens,
                "candidate_count": params.candidate_count,
            }.items():
                if hasattr(GenerationConfig, key):
                    config_kwargs[key] = value
            
            # Add stop sequences if provided and supported
            if params.stop_sequences and hasattr(GenerationConfig, "stop_sequences"):
                config_kwargs["stop_sequences"] = list(params.stop_sequences)

            # Set random seed if provided and supported
            if params.random_seed is not None and hasattr(GenerationConfig, "random_seed"):
                config_kwargs["random_seed"] = params.random_seed
                
            self.logger.debug(f"Using generation config parameters: {config_kwargs}")
            
            return GenerationConfig(**config_kwargs)
        except Exception as e:
            self.logger.warning(f"Error creating generation config with parameters {config_kwargs}: {e}")
            self.logger.warning("Falling back to default GenerationConfig")
            return GenerationConfig()

    @timer("llm_generate")
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        parameters: Optional[GenerationParameters] = None,
        retry_on_error: bool = True,
        use_cache: bool = True,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a response using Vertex AI.

        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt/instructions
            parameters: Generation parameters to use
            retry_on_error: Whether to retry on transient errors
            use_cache: Whether to use response caching
            **kwargs: Additional parameters to pass to the model

        Returns:
            A tuple containing (generated_text, response_metadata)
        """
        self.logger.debug(f"Sending prompt to {self.model}: {prompt[:100]}...")

        # Combine parameters from various sources (kwargs take precedence)
        if parameters is None:
            # Create from kwargs or defaults
            param_dict = {
                "temperature": kwargs.get(
                    "temperature", self.default_parameters.temperature
                ),
                "top_p": kwargs.get("top_p", self.default_parameters.top_p),
                "top_k": kwargs.get("top_k", self.default_parameters.top_k),
                "max_tokens": kwargs.get(
                    "max_tokens", self.default_parameters.max_tokens
                ),
                "candidate_count": kwargs.get(
                    "candidate_count", self.default_parameters.candidate_count
                ),
                "stop_sequences": kwargs.get(
                    "stop_sequences", self.default_parameters.stop_sequences
                ),
            }

            if "random_seed" in kwargs:
                param_dict["random_seed"] = kwargs["random_seed"]

            parameters = GenerationParameters(**param_dict)

        # Create the generation config
        generation_config = self._create_generation_config(parameters)
        
        # Check cache if enabled
        cache_hit = False
        if self.use_cache and use_cache:
            PerformanceTimer.start_timer("llm_cache_lookup")
            cache_key = self._get_cache_key(prompt, system_prompt, parameters.to_dict())
            cached_response = self.response_cache.get(cache_key)
            PerformanceTimer.end_timer("llm_cache_lookup")
            
            if cached_response:
                self.logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                PerformanceTimer.start_timer("llm_cache_hit")
                result, metadata = cached_response
                PerformanceTimer.end_timer("llm_cache_hit")
                
                # Add cache hit info to metadata
                metadata["cache_hit"] = True
                
                # Estimate token counts for recording even for cache hits
                input_tokens = self._estimate_token_count(prompt)
                if system_prompt:
                    input_tokens += self._estimate_token_count(system_prompt)
                output_tokens = metadata.get("token_count", {}).get("output", self._estimate_token_count(result))
                
                # Record in performance timer as a cache hit
                PerformanceTimer.record_tokens("llm_cache_hit", input_tokens, output_tokens)
                
                return result, metadata

        # Apply retry logic if enabled and libraries are available
        generate_func = (
            self._generate_with_retry
            if retry_on_error and HAS_RETRY_LIBS
            else self._generate
        )

        # Generate response with rate limiting
        try:
            # Acquire rate limiter
            with self.rate_limiter:
                # Track timing
                start_time = time.time()
                
                # Start API call timer
                PerformanceTimer.start_timer("llm_api_call")

                # Generate response
                result, response_obj = generate_func(
                    prompt, system_prompt, generation_config
                )
                
                # End API call timer
                PerformanceTimer.end_timer("llm_api_call")

                # Calculate generation time
                generation_time = time.time() - start_time

                # Collect metadata
                metadata = self._create_metadata(
                    generation_config, generation_time, response_obj
                )
                
                # Estimate token counts
                input_tokens = self._estimate_token_count(prompt)
                if system_prompt:
                    input_tokens += self._estimate_token_count(system_prompt)
                output_tokens = self._estimate_token_count(result)
                
                # Try to get accurate token counts from response if available
                if hasattr(response_obj, "usage_metadata"):
                    try:
                        usage = response_obj.usage_metadata
                        if hasattr(usage, "prompt_token_count"):
                            input_tokens = usage.prompt_token_count
                        if hasattr(usage, "candidates_token_count"):
                            output_tokens = usage.candidates_token_count
                    except Exception as e:
                        self.logger.warning(f"Failed to extract token counts from response: {e}")
                
                # Record token usage
                PerformanceTimer.record_tokens("llm_generate", input_tokens, output_tokens)
                
                # Add token counts to metadata
                metadata["token_count"] = {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens
                }
                
                # Cache the result if enabled
                if self.use_cache and use_cache:
                    PerformanceTimer.start_timer("llm_cache_store")
                    cache_key = self._get_cache_key(prompt, system_prompt, parameters.to_dict())
                    self.response_cache.put(cache_key, (result, metadata))
                    PerformanceTimer.end_timer("llm_cache_store")

                self.logger.debug(f"Received response from {self.model}: {result[:100]}...")
                return result, metadata

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")

            # Return error metadata
            error_metadata = {
                "model": self.model,
                "provider": self.provider_name,
                "error": str(e),
                "error_type": type(e).__name__,
                "generation_config": {},  # Empty dict in case attributes don't exist
                "prompt_length": len(prompt),
            }
            
            # Try to safely extract config parameters
            try:
                if hasattr(generation_config, "temperature"):
                    error_metadata["generation_config"]["temperature"] = generation_config.temperature
                if hasattr(generation_config, "top_p"):
                    error_metadata["generation_config"]["top_p"] = generation_config.top_p
                if hasattr(generation_config, "top_k"):
                    error_metadata["generation_config"]["top_k"] = generation_config.top_k
                if hasattr(generation_config, "max_output_tokens"):
                    error_metadata["generation_config"]["max_tokens"] = generation_config.max_output_tokens
            except Exception as config_err:
                self.logger.warning(f"Failed to extract generation config for error metadata: {config_err}")

            # Re-raise the exception
            raise

    def _generate(
        self,
        prompt: str,
        system_prompt: Optional[str],
        generation_config: GenerationConfig,
    ) -> Tuple[str, Any]:
        """Generate a response without retry logic.

        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt/instructions
            generation_config: The generation configuration

        Returns:
            A tuple containing (generated_text, response_object)
        """
        if system_prompt:
            # For models that don't support system_instructions directly,
            # prepend it to the prompt
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            response = self.client.generate_content(
                full_prompt, generation_config=generation_config
            )
        else:
            # Generate response without system prompt
            response = self.client.generate_content(
                prompt, generation_config=generation_config
            )

        return response.text, response

    def _generate_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str],
        generation_config: GenerationConfig,
    ) -> Tuple[str, Any]:
        """Generate a response with retry logic.

        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt/instructions
            generation_config: The generation configuration

        Returns:
            A tuple containing (generated_text, response_object)
        """

        # Define retry decorator
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(
                (
                    ConnectionError,
                    TimeoutError,
                    aiplatform_errors.ResourceExhausted,
                    aiplatform_errors.ServiceUnavailable,
                )
            ),
        )
        def generate_with_retry():
            return self._generate(prompt, system_prompt, generation_config)

        return generate_with_retry()

    def _estimate_token_count(self, text: str) -> int:
        """Estimate the number of tokens in a text.
        
        This is a simple estimation based on whitespace splitting.
        For accurate token counting, a proper tokenizer should be used.
        
        Args:
            text: The text to estimate token count for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
            
        # Simple whitespace-based estimation (rough approximation)
        # A better approach would be to use the actual tokenizer
        # but this is a reasonable approximation for performance metrics
        words = text.split()
        
        # Based on GPT-3 average of ~1.3 tokens per word
        return int(len(words) * 1.3) + 2  # Adding 2 for safety
    
    def _create_metadata(
        self,
        generation_config: GenerationConfig,
        generation_time: float,
        response_obj: Any,
    ) -> Dict[str, Any]:
        """Create metadata from a generation response.

        Args:
            generation_config: The generation configuration used
            generation_time: Time taken for generation in seconds
            response_obj: The response object from Vertex AI

        Returns:
            Response metadata
        """
        metadata = {
            "model": self.model,
            "provider": self.provider_name,
            "generation_time": generation_time,
            "generation_config": {},
        }
        
        # Safely extract generation config parameters
        try:
            config_dict = {}
            if hasattr(generation_config, "temperature"):
                config_dict["temperature"] = generation_config.temperature
            if hasattr(generation_config, "top_p"):
                config_dict["top_p"] = generation_config.top_p
            if hasattr(generation_config, "top_k"):
                config_dict["top_k"] = generation_config.top_k
            if hasattr(generation_config, "max_output_tokens"):
                config_dict["max_tokens"] = generation_config.max_output_tokens
            metadata["generation_config"] = config_dict
        except Exception as e:
            self.logger.warning(f"Failed to extract generation config for metadata: {e}")

        # Add usage information if available
        if hasattr(response_obj, "usage_metadata"):
            try:
                usage = response_obj.usage_metadata
                metadata["usage"] = {
                    "prompt_token_count": usage.prompt_token_count,
                    "candidates_token_count": usage.candidates_token_count,
                    "total_token_count": usage.total_token_count,
                }
            except Exception as e:
                self.logger.warning(f"Failed to extract usage metadata: {e}")

        # Add safety information if available
        if hasattr(response_obj, "prompt_feedback"):
            try:
                metadata["safety"] = {
                    "blocked": response_obj.prompt_feedback.block_reason is not None,
                    "block_reason": response_obj.prompt_feedback.block_reason,
                    "safety_ratings": (
                        [
                            {
                                "category": rating.category,
                                "probability": rating.probability,
                            }
                            for rating in response_obj.prompt_feedback.safety_ratings
                        ]
                        if response_obj.prompt_feedback.safety_ratings
                        else []
                    ),
                }
            except Exception as e:
                self.logger.warning(f"Failed to extract safety metadata: {e}")

        return metadata

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.

        Returns:
            A dictionary with model information
        """
        if self._model_info is None:
            self._model_info = {
                "name": self.model,
                "provider": self.provider_name,
                "project_id": self.project_id,
                "location": self.location,
                "endpoint": self.vertex_endpoint,
                "has_image_support": HAS_IMAGE_SUPPORT,
            }

        return self._model_info


@register_provider
class GeminiProvider(VertexAIProvider):
    """Provider for Google's Gemini via Vertex AI."""

    provider_type = ProviderType.VERTEX_AI

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
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        """Initialize the Gemini provider.

        This is a convenience wrapper around VertexAIProvider specifically for Gemini models.

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud location
            model: Gemini model to use (e.g., gemini-1.5-pro)
            credentials_path: Path to service account credentials
            default_temperature: Default temperature for generation
            default_top_p: Default top_p for generation
            default_top_k: Default top_k for generation
            default_max_tokens: Default maximum tokens for generation
            timeout: Timeout for requests in seconds
            max_retries: Maximum number of retries for failed requests
        """
        # Ensure model name has gemini prefix if not already
        if not model.startswith("gemini-"):
            self.logger.warning(
                f"Model name '{model}' doesn't start with 'gemini-', but this is a Gemini provider"
            )

        # Create default parameters
        default_parameters = GenerationParameters(
            temperature=default_temperature,
            top_p=default_top_p,
            top_k=default_top_k,
            max_tokens=default_max_tokens,
        )

        # Initialize the parent class
        super().__init__(
            project_id=project_id,
            location=location,
            model=model,
            credentials_path=credentials_path,
            default_parameters=default_parameters,
            timeout=timeout,
            max_retries=max_retries,
        )


# The register_provider decorator and PROVIDER_REGISTRY are already defined at the top of the file


def create_provider(provider_type: Union[str, ProviderType], **kwargs) -> LLMProvider:
    """Create a provider based on type.

    Args:
        provider_type: The provider type to create
        **kwargs: Additional arguments to pass to the provider constructor

    Returns:
        LLMProvider instance
    """
    # Convert string to enum if needed
    if isinstance(provider_type, str):
        provider_type = ProviderType(provider_type)

    # Get provider class from registry
    if provider_type not in PROVIDER_REGISTRY:
        registered_providers = ", ".join(p.value for p in PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown provider type: {provider_type}. Registered providers: {registered_providers}"
        )

    provider_cls = PROVIDER_REGISTRY[provider_type]
    return provider_cls(**kwargs)


def create_provider_from_config(config: Dict[str, Any]) -> LLMProvider:
    """Create a provider from a configuration dictionary.

    Args:
        config: Dictionary with provider configuration

    Returns:
        LLMProvider instance
    """
    provider_type = config.pop("provider", "mock")
    return create_provider(provider_type, **config)


# Register built-in providers
@register_provider
class MockProvider(LLMProvider):
    """Mock provider for testing without API calls."""

    provider_type = ProviderType.MOCK

    def __init__(
        self,
        model_name: str = "mock-model",
        responses: Optional[Dict[str, str]] = None,
        default_response: str = '{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}',
        default_parameters: Optional[GenerationParameters] = None,
        delay: float = 0.0,
        should_fail: bool = False,
        fail_after: int = 0,
    ):
        """Initialize the mock provider.

        Args:
            model_name: Mock model name
            responses: Dictionary mapping prompt substrings to responses
            default_response: Default response if no match is found
            default_parameters: Default generation parameters
            delay: Artificial delay in seconds (to simulate API latency)
            should_fail: Whether to simulate failures
            fail_after: Number of successful calls before failing
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.responses = responses or {}
        self.default_response = default_response
        self.default_parameters = default_parameters or GenerationParameters()
        self.delay = delay
        self.should_fail = should_fail
        self.fail_after = fail_after
        self.call_count = 0
        self._model_info: Optional[Dict[str, Any]] = None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        parameters: Optional[GenerationParameters] = None,
        retry_on_error: bool = True,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a mock response.

        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt/instructions
            parameters: Generation parameters
            retry_on_error: Whether to retry on errors
            **kwargs: Additional parameters

        Returns:
            A tuple containing (generated_text, response_metadata)
        """
        self.logger.debug(f"Mock provider received prompt: {prompt[:100]}...")
        self.call_count += 1

        # Add artificial delay if specified
        if self.delay > 0:
            time.sleep(self.delay)

        # Simulate failure if configured
        if self.should_fail and self.call_count > self.fail_after:
            if (
                not retry_on_error or self.call_count % 3 == 0
            ):  # Allow retry to succeed every 3rd try
                raise ConnectionError("Mock provider simulated failure")

        # Track generation time
        start_time = time.time()

        # Find a matching response
        result = None
        matched_key = None

        for key, response in self.responses.items():
            if key in prompt:
                result = response
                matched_key = key
                break

        # Use default if no match found
        if result is None:
            result = self.default_response

        # Calculate generation time including any delay
        generation_time = time.time() - start_time

        # Create metadata
        params = parameters or self.default_parameters
        metadata = {
            "model": self.model_name,
            "provider": self.provider_name,
            "generation_time": generation_time,
            "matched_key": matched_key,
            "call_count": self.call_count,
            "generation_config": params.to_dict(),
            "prompt_length": len(prompt),
            "response_length": len(result),
            "usage": {
                "prompt_token_count": len(prompt) // 4,  # Rough approximation
                "completion_token_count": len(result) // 4,
                "total_token_count": (len(prompt) + len(result)) // 4,
            },
        }

        # Include system prompt info if provided
        if system_prompt:
            metadata["has_system_prompt"] = True
            metadata["system_prompt_length"] = len(system_prompt)

        return result, metadata

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the mock model.

        Returns:
            A dictionary with model information
        """
        if self._model_info is None:
            self._model_info = {
                "name": self.model_name,
                "provider": self.provider_name,
                "is_mock": True,
                "response_patterns": list(self.responses.keys()),
                "num_patterns": len(self.responses),
            }

        return self._model_info


# Convenience functions


def get_gemini_provider(model: str = "gemini-1.5-pro", **kwargs) -> GeminiProvider:
    """Create a Gemini provider with the given model.

    Args:
        model: Gemini model to use
        **kwargs: Additional arguments for GeminiProvider

    Returns:
        GeminiProvider instance
    """
    return GeminiProvider(model=model, **kwargs)


def get_mock_provider(**kwargs) -> MockProvider:
    """Create a mock provider for testing.

    Args:
        **kwargs: Arguments for MockProvider

    Returns:
        MockProvider instance
    """
    return MockProvider(**kwargs)


def get_default_provider(**kwargs) -> LLMProvider:
    """Get the default provider based on environment variables.

    This function checks for environment variables to determine which provider to use:
    - If MOCK_PROVIDER is set to a truthy value, it returns a MockProvider
    - Otherwise, it returns a GeminiProvider

    Args:
        **kwargs: Additional arguments for the provider

    Returns:
        LLMProvider instance
    """
    # Check for mock provider flag
    mock_provider = os.getenv("MOCK_PROVIDER", "").lower() in ("true", "1", "yes", "y")

    if mock_provider:
        return get_mock_provider(**kwargs)
    else:
        return get_gemini_provider(**kwargs)
