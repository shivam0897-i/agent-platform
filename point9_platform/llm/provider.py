"""
LLM Provider
============

Multi-provider LLM abstraction via LiteLLM.
"""

import os
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class LLMProvider:
    """
    Unified LLM provider using LiteLLM.
    
    Supports: Gemini, OpenAI, Anthropic, Groq, Mistral, Ollama
    
    Features:
    - Lazy initialization
    - Fallback model support
    - Proper API key mapping
    """
    
    def __init__(self):
        self._initialized = False
        self._litellm = None
    
    def _initialize(self):
        """Lazy initialization of LiteLLM"""
        if self._initialized:
            return
        
        try:
            import litellm
            litellm.set_verbose = False
            
            # Suppress unnecessary logs
            litellm.suppress_debug_info = True
            
            self._litellm = litellm
            self._initialized = True
            
            # Apply API keys from settings
            self._setup_api_keys()
                    
        except ImportError:
            logger.error("LiteLLM not installed. Run: pip install litellm")
            raise
    
    def _setup_api_keys(self):
        """Setup API keys in environment for LiteLLM"""
        from point9_platform.settings.user import UserSettings
        settings = UserSettings()
        
        # LiteLLM key mapping
        # Note: For Gemini, LiteLLM uses GEMINI_API_KEY or GOOGLE_API_KEY
        key_mapping = {
            "GEMINI_API_KEY": settings.GEMINI_API_KEY,
            "GOOGLE_API_KEY": settings.GEMINI_API_KEY,  # Fallback name
            "OPENAI_API_KEY": settings.OPENAI_API_KEY,
            "ANTHROPIC_API_KEY": settings.ANTHROPIC_API_KEY,
            "MISTRAL_API_KEY": settings.MISTRAL_API_KEY,
            "GROQ_API_KEY": settings.GROQ_API_KEY,
        }
        
        for env_var, value in key_mapping.items():
            if value and not os.environ.get(env_var):
                os.environ[env_var] = value
    
    def completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        timeout: int = 60,
        fallback_model: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Make a completion request with optional fallback.
        
        Args:
            messages: List of message dicts with role and content
            model: Model identifier (e.g., "gemini/gemini-2.0-flash")
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            tools: Tool definitions for function calling
            tool_choice: Tool choice strategy ("auto", "none", or specific tool)
            timeout: Request timeout in seconds (default 60)
            fallback_model: Fallback model if primary fails
            
        Returns:
            LiteLLM response object
        """
        self._initialize()
        
        from point9_platform.settings.user import UserSettings
        settings = UserSettings()
        
        model = model or settings.DEFAULT_LLM_MODEL
        fallback = fallback_model or settings.DEFAULT_LLM_MODEL
        temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        
        call_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            **kwargs
        }
        
        if tools:
            call_kwargs["tools"] = tools
            if tool_choice:
                call_kwargs["tool_choice"] = tool_choice
        
        # Try primary model
        try:
            return self._litellm.completion(**call_kwargs)
        except Exception as e:
            logger.warning("Primary model %s failed: %s", model, e)
            
            # Try fallback if different from primary
            if fallback and fallback != model:
                logger.info("Trying fallback model: %s", fallback)
                call_kwargs["model"] = fallback
                try:
                    return self._litellm.completion(**call_kwargs)
                except Exception as fallback_error:
                    logger.error("Fallback model %s also failed: %s", fallback, fallback_error)
                    raise fallback_error
            raise
    
    async def acompletion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Async version of completion"""
        self._initialize()
        
        from point9_platform.settings.user import UserSettings
        settings = UserSettings()
        
        model = model or settings.DEFAULT_LLM_MODEL
        
        return await self._litellm.acompletion(
            model=model,
            messages=messages,
            **kwargs
        )
    
    def supports_function_calling(self, model: str = None) -> bool:
        """Check if model supports function calling"""
        self._initialize()
        
        from point9_platform.settings.user import UserSettings
        settings = UserSettings()
        
        model = model or settings.DEFAULT_LLM_MODEL
        
        try:
            return self._litellm.supports_function_calling(model=model)
        except Exception:
            # Default to True for known models
            return True


# Singleton instance
_llm_provider: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    """Get singleton LLM provider instance."""
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = LLMProvider()
    return _llm_provider

