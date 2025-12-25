"""
Point9 Agent Platform
=====================

Foundation for all Point9 AI agents.

Exports:
    - BaseAgent: Abstract base class for agents
    - tool: Decorator for registering tools
    - ToolRegistry: Auto-discovery tool system
    - UserSettings: Base settings class
    - SYSTEM_SETTINGS: Platform constants
    - Health endpoints for production monitoring
"""

from point9_platform.agent.base import BaseAgent
from point9_platform.tools.decorator import tool
from point9_platform.tools.registry import ToolRegistry
from point9_platform.settings.user import UserSettings
from point9_platform.settings.system import SYSTEM_SETTINGS
from point9_platform.health import get_health_response, get_info_response, create_health_router

__version__ = "1.0.0"
__all__ = [
    "BaseAgent",
    "tool",
    "ToolRegistry",
    "UserSettings",
    "SYSTEM_SETTINGS",
    "get_health_response",
    "get_info_response",
    "create_health_router",
]



