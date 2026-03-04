"""
Tool Decorator
==============

Register functions as agent tools with simple decorator.
"""

from typing import Callable, Dict, Any, Optional, List, get_type_hints
import inspect
import functools
import logging

logger = logging.getLogger(__name__)

# Global registry - populated by @tool decorators
# Now tracks origin module for per-agent filtering
_TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}


def tool(
    name: str,
    description: str,
    parameters: Optional[Dict] = None
) -> Callable:
    """
    Decorator to register a function as an agent tool.
    
    The decorated function should accept:
    1. Tool-specific parameters (validated)
    2. 'state' parameter (current agent state, injected by executor)
    
    Args:
        name: Tool name (used in LLM function calling)
        description: Human-readable description for LLM
        parameters: Optional JSON Schema for parameters.
                   If not provided, auto-generated from type hints.
    
    Example:
        @tool(
            name="extract_cheque_data",
            description="Extract fields from a cheque image"
        )
        def extract_cheque_data(document_id: str, fields: list, state: dict) -> dict:
            document = state["documents"][document_id]
            # ... extraction logic
            return {"status": "success", "data": {...}}
    
    Returns:
        Decorated function (unchanged behavior, but registered)
    """
    def decorator(func: Callable) -> Callable:
        # Auto-generate parameter schema if not provided
        param_schema = parameters or _generate_schema(func)
        
        # Get the module where this tool is defined
        origin_module = func.__module__
        
        # Register the tool with origin tracking
        _TOOL_REGISTRY[name] = {
            "function": func,
            "name": name,
            "description": description,
            "origin_module": origin_module,  # Track where tool came from
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": param_schema
                }
            }
        }
        
        logger.debug("Registered tool: %s from %s", name, origin_module)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def get_all_tools() -> Dict[str, Dict[str, Any]]:
    """Get all registered tools"""
    return _TOOL_REGISTRY.copy()


def get_tools_by_package(package_prefix: str) -> Dict[str, Dict[str, Any]]:
    """
    Get tools that belong to a specific package.
    
    Args:
        package_prefix: Package prefix to filter by (e.g., "sample_agent.tools")
    
    Returns:
        Dict of tools whose origin_module starts with the package prefix
    """
    return {
        name: tool_info
        for name, tool_info in _TOOL_REGISTRY.items()
        if tool_info.get("origin_module", "").startswith(package_prefix)
    }


def get_tool_function(name: str) -> Optional[Callable]:
    """Get tool function by name"""
    tool_info = _TOOL_REGISTRY.get(name)
    return tool_info["function"] if tool_info else None


def get_tool_schemas() -> List[Dict]:
    """Get OpenAI-compatible tool schemas for LLM"""
    return [t["schema"] for t in _TOOL_REGISTRY.values()]


def clear_registry() -> None:
    """Clear tool registry (useful for testing)"""
    _TOOL_REGISTRY.clear()


def _generate_schema(func: Callable) -> Dict[str, Any]:
    """Auto-generate JSON Schema from function signature"""
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
    
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        # Skip 'state' parameter (injected by executor)
        if param_name == "state":
            continue
        
        param_type = hints.get(param_name, str)
        json_type = _python_type_to_json(param_type)
        
        properties[param_name] = {"type": json_type}
        
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    
    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


def _python_type_to_json(py_type) -> str:
    """Convert Python type to JSON Schema type"""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    
    # Handle Optional, List, etc.
    origin = getattr(py_type, "__origin__", None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    
    return type_map.get(py_type, "string")

