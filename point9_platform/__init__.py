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
    - Storage utilities (S3, MongoDB)
"""

from point9_platform.agent.base import BaseAgent
from point9_platform.tools.decorator import tool
from point9_platform.tools.registry import ToolRegistry
from point9_platform.settings.user import UserSettings
from point9_platform.settings.system import SYSTEM_SETTINGS
from point9_platform.health import get_health_response, get_info_response, create_health_router

# Storage utilities (optional — requires boto3/pymongo)
try:
    from point9_platform.storage import S3Storage, get_s3_storage, MongoStore, get_mongo_store
except ImportError:
    S3Storage = None  # type: ignore[assignment,misc]
    get_s3_storage = None  # type: ignore[assignment]
    MongoStore = None  # type: ignore[assignment,misc]
    get_mongo_store = None  # type: ignore[assignment]

# Evaluation framework (optional — requires ragas/evaluate/litellm)
try:
    from point9_platform.evaluation import Evaluator, EvaluationResult, get_evaluator
except ImportError:
    Evaluator = None  # type: ignore[assignment,misc]
    EvaluationResult = None  # type: ignore[assignment,misc]
    get_evaluator = None  # type: ignore[assignment]

__version__ = "1.1.1"
__all__ = [
    "BaseAgent",
    "tool",
    "ToolRegistry",
    "UserSettings",
    "SYSTEM_SETTINGS",
    "get_health_response",
    "get_info_response",
    "create_health_router",
    # Storage
    "S3Storage",
    "get_s3_storage",
    "MongoStore",
    "get_mongo_store",
    # Evaluation
    "Evaluator",
    "EvaluationResult",
    "get_evaluator",
]



