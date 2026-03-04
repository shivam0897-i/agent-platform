"""
Tool Registry
=============

Auto-discovers tools from agent's tools package.
"""

import importlib
import pkgutil
from typing import Dict, List, Callable, Optional
import logging

from point9_platform.tools.decorator import get_tools_by_package

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Manages tool discovery and access for an agent.
    
    Tools are discovered by importing all modules in the specified package,
    which triggers @tool decorators to register functions.
    
    IMPORTANT: Each registry only sees tools from its own package,
    preventing cross-agent tool leakage when multiple agents run
    in the same process.
    
    Usage:
        registry = ToolRegistry("myagent.tools")
        registry.discover()
        
        definitions = registry.get_definitions()  # For LLM
        func = registry.get_function("extract_data")  # For execution
    """
    
    def __init__(self, tools_package: str):
        """
        Initialize registry.
        
        Args:
            tools_package: Python package path (e.g., "agent.tools")
        """
        self.tools_package = tools_package
        self._discovered = False
        self._tool_cache: Dict[str, Dict] = {}
    
    def discover(self) -> None:
        """
        Discover tools by importing all modules in the tools package.
        
        This triggers @tool decorators which register functions.
        Safe to call multiple times (no-op after first discovery).
        
        Only caches tools that belong to THIS agent's package,
        filtering out tools from other agents.
        """
        if self._discovered:
            return
        
        try:
            package = importlib.import_module(self.tools_package)
        except ImportError as e:
            logger.warning("Could not import tools package '%s': %s", self.tools_package, e)
            self._discovered = True
            return
        
        # Get package directory
        if hasattr(package, "__path__"):
            package_path = package.__path__
        else:
            logger.warning("'%s' is not a package", self.tools_package)
            self._discovered = True
            return
        
        # Import all modules in the package
        for module_info in pkgutil.iter_modules(package_path):
            if not module_info.name.startswith("_"):
                module_name = f"{self.tools_package}.{module_info.name}"
                try:
                    importlib.import_module(module_name)
                    logger.debug("Discovered tools from: %s", module_name)
                except Exception as e:
                    logger.error("Error importing %s: %s", module_name, e)
        
        # Cache ONLY tools from this agent's package (isolation!)
        self._tool_cache = get_tools_by_package(self.tools_package)
        self._discovered = True
        
        logger.info("Discovered %s tools from %s", len(self._tool_cache), self.tools_package)
    
    def get_definitions(self) -> List[Dict]:
        """Get OpenAI-compatible tool definitions for LLM function calling"""
        self.discover()
        return [t["schema"] for t in self._tool_cache.values()]
    
    def get_function(self, name: str) -> Optional[Callable]:
        """Get tool function by name"""
        self.discover()
        tool_info = self._tool_cache.get(name)
        return tool_info["function"] if tool_info else None
    
    def get_all_names(self) -> List[str]:
        """Get list of all registered tool names"""
        self.discover()
        return list(self._tool_cache.keys())
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered"""
        self.discover()
        return name in self._tool_cache
    
    def get_tool_info(self, name: str) -> Optional[Dict]:
        """Get full tool info including schema"""
        self.discover()
        return self._tool_cache.get(name)
