"""
Tool Executor
=============

Executes tools with retry logic and observability.
"""

import json
import logging
import time
from typing import Dict, Any, Callable, Optional

from point9_platform.tools.registry import ToolRegistry
from point9_platform.settings.system import SYSTEM_SETTINGS

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Executes tools with:
    - Retry logic with exponential backoff
    - Step emission for observability
    - State injection
    """
    
    def __init__(self, state: Dict[str, Any], tool_registry: ToolRegistry):
        """
        Initialize executor.
        
        Args:
            state: Current agent state (for context injection)
            tool_registry: Registry containing tool functions
        """
        self.state = state
        self.tool_registry = tool_registry
    
    def get_tool_definitions(self) -> list:
        """Get tool definitions for LLM function calling"""
        return self.tool_registry.get_definitions()
    
    def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        emitter=None
    ) -> Dict[str, Any]:
        """
        Execute a tool with retry logic.
        
        Args:
            tool_name: Name of the tool to execute
            args: Tool arguments
            emitter: Optional StepEmitter for progress tracking
            
        Returns:
            Tool output as dictionary
        """
        tool_fn = self.tool_registry.get_function(tool_name)
        if not tool_fn:
            return {"status": "failed", "error": f"Unknown tool: {tool_name}"}
        
        # Emit tool start
        if emitter:
            from point9_platform.observability.emitter import StepType, StepStatus
            emitter.emit_blocking(
                StepType.TOOL_CALL,
                f"Executing {tool_name}...",
                status=StepStatus.IN_PROGRESS,
                details={"tool": tool_name, "args": _sanitize_args(args)}
            )
        
        # Retry loop
        last_error = None
        for attempt in range(SYSTEM_SETTINGS.TOOL_RETRY_ATTEMPTS):
            try:
                # Execute tool with state injection
                result = tool_fn(**args, state=self.state)
                
                # Convert Pydantic model to dict if needed
                if hasattr(result, 'model_dump'):
                    result = result.model_dump()
                
                # Emit success
                if emitter:
                    from point9_platform.observability.emitter import StepType, StepStatus
                    emitter.emit_blocking(
                        StepType.TOOL_RESULT,
                        f"{tool_name} completed",
                        status=StepStatus.COMPLETED,
                        details={"status": result.get("status", "unknown")}
                    )
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Tool {tool_name} attempt {attempt + 1} failed: {e}")
                
                if attempt < SYSTEM_SETTINGS.TOOL_RETRY_ATTEMPTS - 1:
                    # Exponential backoff
                    delay = SYSTEM_SETTINGS.TOOL_RETRY_DELAY * (2 ** attempt)
                    time.sleep(delay)
        
        # All retries failed
        logger.error(f"Tool {tool_name} failed after {SYSTEM_SETTINGS.TOOL_RETRY_ATTEMPTS} attempts")
        
        if emitter:
            from point9_platform.observability.emitter import StepType, StepStatus
            emitter.emit_blocking(
                StepType.ERROR,
                f"{tool_name} failed: {str(last_error)[:50]}",
                status=StepStatus.FAILED
            )
        
        return {
            "status": "failed",
            "document_id": args.get("document_id", "unknown"),
            "error": str(last_error)
        }


def _sanitize_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive data from args for logging"""
    sanitized = dict(args)
    sensitive_keys = ["api_key", "password", "token", "secret"]
    for key in sensitive_keys:
        if key in sanitized:
            sanitized[key] = "***"
    return sanitized
