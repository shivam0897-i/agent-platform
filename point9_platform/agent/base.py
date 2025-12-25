"""
Abstract Base Agent
===================

Foundation class for all Point9 agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, TypeVar, Generic, Optional
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

StateT = TypeVar('StateT', bound=Dict[str, Any])


class BaseAgent(ABC, Generic[StateT]):
    """
    Abstract Base Class for all Point9 agents.
    
    MUST IMPLEMENT (Abstract Methods):
        - get_agent_name() → str
        - get_domain_keywords() → List[str]
        - create_initial_state(session_id) → StateT
        - get_prompts() → Dict[str, str]
    
    MAY OVERRIDE (Virtual Methods):
        - create_graph() → Override for custom workflow
        - create_planner_node() → Custom planning logic
        - create_executor_node() → Custom execution logic
        - create_reflector_node() → Custom reflection logic
        - create_responder_node() → Custom response generation
        - on_start() → Lifecycle hook
        - on_complete() → Lifecycle hook
        - on_error() → Lifecycle hook
    
    Example:
        class ChequeAgent(BaseAgent[ChequeState]):
            def get_agent_name(self):
                return "cheque_processor"
            
            def get_domain_keywords(self):
                return ["cheque", "check", "MICR"]
            
            def create_initial_state(self, session_id):
                return ChequeState(session_id=session_id, ...)
            
            def get_prompts(self):
                return {"planner": "...", "executor": "..."}
    """
    
    def __init__(
        self,
        session_id: str,
        tools_package: str,
        settings: Optional["UserSettings"] = None,
        llm: Optional["LLMProvider"] = None
    ):
        """
        Initialize the agent.
        
        Args:
            session_id: Unique session identifier
            tools_package: Python package path containing tools (e.g., "agent.tools")
            settings: User settings instance (optional, uses defaults if None)
            llm: LLM provider instance (optional, creates default if None)
        """
        from point9_platform.settings.user import UserSettings
        from point9_platform.llm.provider import get_llm_provider
        from point9_platform.observability.emitter import get_or_create_emitter
        from point9_platform.tools.registry import ToolRegistry
        
        self.session_id = session_id
        self.settings = settings or UserSettings()
        self.llm = llm or get_llm_provider()
        self.emitter = get_or_create_emitter(session_id)
        
        # Initialize tool registry and discover tools
        self.tool_registry = ToolRegistry(tools_package)
        self.tool_registry.discover()
        
        # Graph is lazily initialized
        self._graph = None
        self._compiled_graph = None
    
    # =========================================================================
    # ABSTRACT METHODS - MUST IMPLEMENT
    # =========================================================================
    
    @abstractmethod
    def get_agent_name(self) -> str:
        """Return unique agent identifier (e.g., 'cheque_processor')"""
        pass
    
    @abstractmethod
    def get_domain_keywords(self) -> List[str]:
        """Return keywords for domain validation"""
        pass
    
    @abstractmethod
    def create_initial_state(self, session_id: str) -> StateT:
        """Create initial state for a new session"""
        pass
    
    @abstractmethod
    def get_prompts(self) -> Dict[str, str]:
        """
        Return prompt templates.
        
        Expected keys: 'planner', 'executor', 'reflector', 'responder'
        """
        pass
    
    # =========================================================================
    # CONCRETE METHODS - SHARED BEHAVIOR
    # =========================================================================
    
    @property
    def graph(self):
        """Get compiled graph (lazy initialization)"""
        if self._compiled_graph is None:
            self._compiled_graph = self.create_graph()
        return self._compiled_graph
    
    def process(
        self,
        message: str,
        documents: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main entry point - process a user request.
        
        Args:
            message: User's message
            documents: Optional dict of documents to process
            **kwargs: Additional arguments passed to state
        
        Returns:
            Dict with success status, message, and results
        """
        # Validate domain
        if not self.validate_domain(message):
            return {
                "success": False,
                "error": f"Request outside {self.get_agent_name()} domain",
                "session_id": self.session_id
            }
        
        # Create initial state
        state = self.create_initial_state(self.session_id)
        state["messages"] = [{"role": "user", "content": message}]
        
        if documents:
            state["documents"] = documents
        
        # Lifecycle hook
        self.on_start(state)
        
        try:
            # Run the graph
            config = {"configurable": {"thread_id": self.session_id}}
            final_state = self.graph.invoke(state, config)
            
            # Extract result
            result = self._extract_result(final_state)
            
            # Lifecycle hook
            self.on_complete(state, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Agent {self.get_agent_name()} error: {e}")
            self.on_error(state, e)
            return {
                "success": False,
                "error": str(e),
                "session_id": self.session_id
            }
    
    def validate_domain(self, message: str) -> bool:
        """Check if message is within agent's domain"""
        keywords = self.get_domain_keywords()
        message_lower = message.lower()
        return any(kw.lower() in message_lower for kw in keywords)
    
    def create_graph(self):
        """
        Build the LangGraph workflow.
        
        Default: Planner → Executor ↔ Reflector → Responder
        
        Override for custom workflows.
        """
        from point9_platform.graph.builder import build_default_graph
        return build_default_graph(
            agent=self,
            planner=self.create_planner_node(),
            executor=self.create_executor_node(),
            reflector=self.create_reflector_node(),
            responder=self.create_responder_node()
        )
    
    # =========================================================================
    # NODE FACTORIES - OVERRIDE FOR CUSTOM BEHAVIOR
    # =========================================================================
    
    def create_planner_node(self):
        """Create the planner node (override for custom planning)"""
        from point9_platform.graph.nodes import create_default_planner
        return create_default_planner(self.llm, self.get_prompts().get("planner"))
    
    def create_executor_node(self):
        """Create the executor node (override for custom execution)"""
        from point9_platform.graph.nodes import create_default_executor
        return create_default_executor(self.llm, self.tool_registry)
    
    def create_reflector_node(self):
        """Create the reflector node (override for custom reflection)"""
        from point9_platform.graph.nodes import create_default_reflector
        return create_default_reflector(self.llm)
    
    def create_responder_node(self):
        """Create the responder node (override for custom responses)"""
        from point9_platform.graph.nodes import create_default_responder
        return create_default_responder(self.llm, self.get_prompts().get("responder"))
    
    # =========================================================================
    # LIFECYCLE HOOKS - OPTIONAL OVERRIDE
    # =========================================================================
    
    def on_start(self, state: StateT) -> None:
        """Called before processing starts. Override for custom setup."""
        from point9_platform.observability.emitter import StepType, StepStatus
        self.emitter.emit_blocking(
            step_type=StepType.AGENT_PLANNING,
            message=f"{self.get_agent_name()} processing started",
            status=StepStatus.IN_PROGRESS
        )
    
    def on_complete(self, state: StateT, result: Dict) -> None:
        """Called after successful processing. Override for cleanup."""
        self.emitter.complete_blocking(success=True)
    
    def on_error(self, state: StateT, error: Exception) -> None:
        """Called when processing fails. Override for error handling."""
        self.emitter.complete_blocking(success=False, final_message=str(error))
    
    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================
    
    def _extract_result(self, final_state: Dict) -> Dict[str, Any]:
        """Extract response from final state"""
        response_message = ""
        for msg in reversed(final_state.get("messages", [])):
            if msg.get("role") == "assistant":
                response_message = msg.get("content", "")
                break
        
        return {
            "success": True,
            "message": response_message,
            "results": final_state.get("results", {}),
            "session_id": self.session_id
        }
