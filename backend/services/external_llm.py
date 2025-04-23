import os
import json
import logging
import time
from typing import Dict, Any, List, Optional

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger("ai_assistant.external_llm")

# API keys and configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# Check for package availability (don't fail if not available)
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not available. OpenAI integration disabled.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic package not available. Claude integration disabled.")

try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logger.warning("HuggingFace Hub package not available. HuggingFace integration disabled.")

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    logger.warning("Cohere package not available. Cohere integration disabled.")


class ExternalLLMService:
    """
    Service for interacting with external Large Language Models (LLMs)
    through their APIs. Supports multiple providers and fallback mechanisms.
    """
    
    def __init__(self, 
                 default_provider: str = "openai", 
                 default_model: Optional[str] = None,
                 api_keys: Optional[Dict[str, str]] = None,
                 max_retries: int = 3,
                 request_timeout: int = 60,
                 verbose: bool = False):
        """
        Initialize the external LLM service.
        
        Args:
            default_provider: Default LLM provider to use ('openai', 'anthropic', 'huggingface', 'cohere')
            default_model: Default model for the provider (if None, uses service default)
            api_keys: Dictionary of API keys for different services
            max_retries: Maximum number of retries for failed requests
            request_timeout: Timeout for API requests in seconds
            verbose: Whether to log verbose information
        """
        self.default_provider = default_provider
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.verbose = verbose
        
        # Set default models based on provider
        self.default_models = {
            "openai": default_model or "gpt-3.5-turbo", 
            "anthropic": default_model or "claude-3-haiku-20240307",
            "huggingface": default_model or "mistralai/mistral-7b-instruct",
            "cohere": default_model or "command-light"
        }
        
        # Store API keys
        self.api_keys = api_keys or {
            "openai": OPENAI_API_KEY,
            "anthropic": ANTHROPIC_AVAILABLE,
            "huggingface": HUGGINGFACE_API_KEY,
            "cohere": COHERE_API_KEY
        }
        
        # Initialize clients if available
        self.clients = {}
        self._initialize_clients()
        
        # Verify we have at least one working client
        if not self.clients:
            logger.warning("No LLM clients could be initialized. External LLM functionality will be limited.")

    def _initialize_clients(self):
        """Initialize API clients for available services"""
        # OpenAI
        if OPENAI_AVAILABLE and self.api_keys.get("openai"):
            try:
                self.clients["openai"] = OpenAI(api_key=self.api_keys["openai"])
                logger.info("OpenAI client initialized successfully.")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {str(e)}")
        
        # Anthropic
        if ANTHROPIC_AVAILABLE and self.api_keys.get("anthropic"):
            try:
                self.clients["anthropic"] = anthropic.Anthropic(api_key=self.api_keys["anthropic"])
                logger.info("Anthropic client initialized successfully.")
            except Exception as e:
                logger.error(f"Error initializing Anthropic client: {str(e)}")
        
        # HuggingFace
        if HUGGINGFACE_AVAILABLE and self.api_keys.get("huggingface"):
            try:
                self.clients["huggingface"] = InferenceClient(token=self.api_keys["huggingface"])
                logger.info("HuggingFace client initialized successfully.")
            except Exception as e:
                logger.error(f"Error initializing HuggingFace client: {str(e)}")
        
        # Cohere
        if COHERE_AVAILABLE and self.api_keys.get("cohere"):
            try:
                self.clients["cohere"] = cohere.Client(api_key=self.api_keys["cohere"])
                logger.info("Cohere client initialized successfully.")
            except Exception as e:
                logger.error(f"Error initializing Cohere client: {str(e)}")

    def get_available_providers(self) -> List[str]:
        """
        Get a list of available LLM providers.
        
        Returns:
            List of available provider names
        """
        return list(self.clients.keys())

    def get_default_model(self, provider: Optional[str] = None) -> str:
        """
        Get the default model for a provider.
        
        Args:
            provider: Provider to get default model for (uses default_provider if None)
            
        Returns:
            Default model name for the provider
        """
        provider = provider or self.default_provider
        return self.default_models.get(provider, "")

    async def generate_text(self, 
                    prompt: str, 
                    provider: Optional[str] = None, 
                    model: Optional[str] = None,
                    max_tokens: int = 500,
                    temperature: float = 0.7,
                    **kwargs) -> Dict[str, Any]:
        """
        Generate text using an external LLM.
        
        Args:
            prompt: Text prompt to send to the LLM
            provider: LLM provider to use (default_provider if None)
            model: Model to use (provider default if None)
            max_tokens: Maximum tokens in the response
            temperature: Temperature for sampling (0.0 to 1.0) - higher is more random
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dictionary containing:
                - text: Generated text
                - provider: Provider used
                - model: Model used
                - usage: Token usage information if available
                - error: Error message if an error occurred
        """
        provider = provider or self.default_provider
        model = model or self.default_models.get(provider)
        
        if not model:
            return {
                "error": f"No model specified for provider {provider}",
                "text": "",
                "provider": provider,
                "model": None
            }
        
        if provider not in self.clients:
            available = ", ".join(self.clients.keys()) if self.clients else "none"
            return {
                "error": f"Provider {provider} not available. Available providers: {available}",
                "text": "",
                "provider": provider,
                "model": model
            }
            
        # Try to generate with the specified provider
        for attempt in range(self.max_retries):
            try:
                if provider == "openai":
                    return await self._generate_openai(prompt, model, max_tokens, temperature, **kwargs)
                elif provider == "anthropic":
                    return await self._generate_anthropic(prompt, model, max_tokens, temperature, **kwargs)
                elif provider == "huggingface":
                    return await self._generate_huggingface(prompt, model, max_tokens, temperature, **kwargs)
                elif provider == "cohere":
                    return await self._generate_cohere(prompt, model, max_tokens, temperature, **kwargs)
                else:
                    return {
                        "error": f"Unknown provider: {provider}",
                        "text": "",
                        "provider": provider,
                        "model": model
                    }
            except Exception as e:
                logger.error(f"Error generating text with {provider} (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    return {
                        "error": f"Failed to generate text after {self.max_retries} attempts: {str(e)}",
                        "text": "",
                        "provider": provider,
                        "model": model
                    }
                time.sleep(2 ** attempt)  # Exponential backoff
        
    async def _generate_openai(self, prompt, model, max_tokens, temperature, **kwargs):
        """Generate text using OpenAI API"""
        if self.verbose:
            logger.info(f"Generating text with OpenAI model {model}")
            
        messages = kwargs.get("messages", [{"role": "user", "content": prompt}])
        if not kwargs.get("messages") and prompt:
            messages = [{"role": "user", "content": prompt}]
            
        try:
            response = self.clients["openai"].chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **{k: v for k, v in kwargs.items() if k not in ["messages"]}
            )
            
            return {
                "text": response.choices[0].message.content,
                "provider": "openai",
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            }
        except Exception as e:
            raise Exception(f"OpenAI generation error: {str(e)}")
    
    async def _generate_anthropic(self, prompt, model, max_tokens, temperature, **kwargs):
        """Generate text using Anthropic API"""
        if self.verbose:
            logger.info(f"Generating text with Anthropic model {model}")
            
        system = kwargs.get("system", "")
        
        try:
            response = self.clients["anthropic"].messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **{k: v for k, v in kwargs.items() if k not in ["system"]}
            )
            
            return {
                "text": response.content[0].text,
                "provider": "anthropic",
                "model": model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            }
        except Exception as e:
            raise Exception(f"Anthropic generation error: {str(e)}")
    
    async def _generate_huggingface(self, prompt, model, max_tokens, temperature, **kwargs):
        """Generate text using HuggingFace API"""
        if self.verbose:
            logger.info(f"Generating text with HuggingFace model {model}")
            
        try:
            parameters = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": kwargs.get("return_full_text", False),
            }
            
            # Add extra parameters that HuggingFace supports
            for param in ["top_p", "repetition_penalty", "top_k"]:
                if param in kwargs:
                    parameters[param] = kwargs[param]
            
            response = self.clients["huggingface"].text_generation(
                prompt,
                model=model,
                **parameters
            )
            
            return {
                "text": response,
                "provider": "huggingface",
                "model": model,
            }
        except Exception as e:
            raise Exception(f"HuggingFace generation error: {str(e)}")
    
    async def _generate_cohere(self, prompt, model, max_tokens, temperature, **kwargs):
        """Generate text using Cohere API"""
        if self.verbose:
            logger.info(f"Generating text with Cohere model {model}")
            
        try:
            response = self.clients["cohere"].generate(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                **{k: v for k, v in kwargs.items() if k in [
                    "p", "k", "frequency_penalty", "presence_penalty"
                ]}
            )
            
            return {
                "text": response.generations[0].text,
                "provider": "cohere",
                "model": model,
                "usage": {
                    "total_tokens": response.meta.billed_units.total_tokens,
                }
            }
        except Exception as e:
            raise Exception(f"Cohere generation error: {str(e)}")
    
    def list_available_models(self, provider: Optional[str] = None) -> List[str]:
        """
        List available models for a provider.
        
        Args:
            provider: Provider to list models for (all providers if None)
            
        Returns:
            List of available model names
        """
        models = []
        
        if provider:
            providers = [provider] if provider in self.clients else []
        else:
            providers = self.clients.keys()
            
        for p in providers:
            if p == "openai" and OPENAI_AVAILABLE:
                try:
                    response = self.clients["openai"].models.list()
                    models.extend([f"openai:{m.id}" for m in response.data])
                except Exception as e:
                    logger.error(f"Error listing OpenAI models: {str(e)}")
            elif p == "anthropic" and ANTHROPIC_AVAILABLE:
                # Anthropic doesn't have a list models API, so we provide common ones
                models.extend([
                    "anthropic:claude-3-opus-20240229",
                    "anthropic:claude-3-sonnet-20240229",
                    "anthropic:claude-3-haiku-20240307",
                    "anthropic:claude-2.1",
                    "anthropic:claude-2.0",
                    "anthropic:claude-instant-1.2"
                ])
            elif p == "huggingface" and HUGGINGFACE_AVAILABLE:
                # Add common HF models - there are too many to list via API
                models.extend([
                    "huggingface:mistralai/mistral-7b-instruct",
                    "huggingface:meta-llama/Llama-2-7b-chat-hf",
                    "huggingface:meta-llama/Llama-2-13b-chat-hf", 
                    "huggingface:tiiuae/falcon-7b-instruct",
                    "huggingface:bigcode/starcoder"
                ])
            elif p == "cohere" and COHERE_AVAILABLE:
                models.extend([
                    "cohere:command",
                    "cohere:command-light",
                    "cohere:command-r",
                    "cohere:command-r-plus"
                ])
                
        return models

# Create a singleton instance
external_llm_service = ExternalLLMService()
