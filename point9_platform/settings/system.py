"""
System Settings
===============

Immutable platform-level constants.
These values are FIXED and should NOT be overridden by users.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemSettings:
    """
    Platform-level constants.
    
    These values ensure consistent behavior across all agents.
    To change these, update the platform and release a new version.
    
    DO NOT modify these values per-agent or per-deployment.
    """
    
    # Execution limits
    MAX_ITERATIONS: int = 10
    MAX_MESSAGES: int = 10
    
    # Retry configuration
    TOOL_RETRY_ATTEMPTS: int = 3
    TOOL_RETRY_DELAY: float = 1.0  # Base delay (exponential backoff)
    
    # Timeouts (seconds)
    LLM_TIMEOUT: int = 60
    TOOL_TIMEOUT: int = 30
    
    # Version info
    PLATFORM_VERSION: str = "1.1.2"


# Singleton instance - import this
SYSTEM_SETTINGS = SystemSettings()
