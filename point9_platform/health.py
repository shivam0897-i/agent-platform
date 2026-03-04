"""
Health & Info Endpoints
=======================

Standard endpoints for production monitoring.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List
import platform
import os

from point9_platform.settings.system import SYSTEM_SETTINGS


_start_time = datetime.now(timezone.utc)


def get_health_response() -> Dict[str, Any]:
    """
    Get health check response.
    
    Returns:
        Health status dict
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }


def get_info_response(
    agent_name: str,
    agent_version: str = "1.0.0",
    tools: List[str] = None
) -> Dict[str, Any]:
    """
    Get agent info response.
    
    Args:
        agent_name: Name of the agent
        agent_version: Version of the agent
        tools: List of registered tool names
    
    Returns:
        Agent info dict
    """
    uptime = datetime.now(timezone.utc) - _start_time
    
    return {
        "agent": agent_name,
        "version": agent_version,
        "platform_version": SYSTEM_SETTINGS.PLATFORM_VERSION,
        "uptime_seconds": int(uptime.total_seconds()),
        "tools": tools or [],
        "python_version": platform.python_version(),
        "environment": os.getenv("ENVIRONMENT", "development")
    }


def get_ready_response(checks: Dict[str, bool] = None) -> Dict[str, Any]:
    """
    Get readiness check response.
    
    Args:
        checks: Dict of check name → pass/fail
    
    Returns:
        Readiness status dict
    """
    checks = checks or {"default": True}
    all_ready = all(checks.values())
    
    return {
        "ready": all_ready,
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }


# =============================================================================
# FastAPI Router (optional integration)
# =============================================================================

def create_health_router():
    """
    Create FastAPI router with health endpoints.
    
    Usage:
        from point9_platform.health import create_health_router
        app.include_router(create_health_router())
    """
    try:
        from fastapi import APIRouter
    except ImportError:
        return None
    
    router = APIRouter(tags=["health"])
    
    @router.get("/health")
    async def health():
        return get_health_response()
    
    @router.get("/ready")
    async def ready():
        return get_ready_response()
    
    return router
