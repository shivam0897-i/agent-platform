"""
User Settings
=============

Base class for user-configurable settings.
Supports loading from: YAML config file, .env file, and environment variables.

Priority (highest to lowest):
1. Environment variables
2. .env file
3. config.yaml file
4. Default values

Agents should EXTEND this class with domain-specific settings.
"""

import os
import yaml
import logging
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load settings from YAML config file.
    
    Args:
        config_path: Path to YAML file (default: config.yaml in cwd)
    
    Returns:
        Dict of settings from YAML, or empty dict if file not found
    """
    path = Path(config_path)
    
    if not path.exists():
        # Try in current working directory
        path = Path.cwd() / config_path
    
    if not path.exists():
        logger.debug("Config file not found: %s", config_path)
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
            logger.info("Loaded config from: %s", path)
            return config
    except Exception as e:
        logger.warning("Error loading %s: %s", config_path, e)
        return {}


class UserSettings(BaseSettings):
    """
    Base class for user-configurable settings.
    
    Values are loaded in this order (later overrides earlier):
    1. Default values (defined in class)
    2. config.yaml file
    3. .env file
    4. Environment variables
    
    Example - Agent extending with domain-specific settings:
        class ChequeSettings(UserSettings):
            MICR_CONFIDENCE_THRESHOLD: float = 0.85
            BANK_CODE_VALIDATION: bool = True
            
            class Config(UserSettings.Config):
                yaml_file = "cheque_config.yaml"  # Custom YAML file
    """
    
    # === AGENT IDENTITY ===
    AGENT_NAME: str = "Agent"
    AGENT_DESCRIPTION: str = "AI Agent"
    
    # === LLM CONFIGURATION ===
    DEFAULT_LLM_MODEL: str = "gemini/gemini-2.0-flash"
    DEFAULT_VISION_MODEL: str = "gemini/gemini-2.0-flash"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 4096
    
    # === API KEYS ===
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    MISTRAL_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    
    def __init__(self, **kwargs):
        # Load YAML config first
        yaml_file = getattr(self.Config, 'yaml_file', 'config.yaml')
        yaml_config = load_yaml_config(yaml_file)
        
        # Merge: YAML values (lower priority) with kwargs (higher priority)
        merged = {**yaml_config, **kwargs}
        
        super().__init__(**merged)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore unknown fields
        yaml_file = "config.yaml"  # Default YAML config file


# =============================================================================
# Cached settings accessor
# =============================================================================

_settings_cache: Optional[UserSettings] = None


def get_user_settings() -> UserSettings:
    """Get cached user settings instance"""
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = UserSettings()
    return _settings_cache


def clear_settings_cache() -> None:
    """Clear settings cache (for testing)"""
    global _settings_cache
    _settings_cache = None


def reload_settings() -> UserSettings:
    """Force reload settings from files"""
    clear_settings_cache()
    return get_user_settings()
