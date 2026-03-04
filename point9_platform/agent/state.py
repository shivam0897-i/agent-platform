"""
Base Agent State
================

Minimal state schema that all agents must have.

IMPORTANT NOTES:
- BaseAgentState includes BOTH core fields AND planning fields required by
  default graph nodes (planner, executor, reflector, responder).
- If you override create_graph() with custom nodes that don't use planning
  fields, you can ignore plan/current_step/current_task/thoughts/results.
- The message_reducer auto-truncates history to MAX_MESSAGES (default 10).
- Extend this class with domain-specific fields using TypedDict inheritance:
    class MyState(BaseAgentState):
        my_field: str
"""

from typing import Dict, Any, List, Optional, Annotated, TypedDict


def message_reducer(current: List[Dict], new: List[Dict]) -> List[Dict]:
    """
    Custom reducer that truncates message history to prevent token explosion.
    """
    from point9_platform.settings.system import SYSTEM_SETTINGS
    combined = current + new
    return combined[-SYSTEM_SETTINGS.MAX_MESSAGES:]


class BaseAgentState(TypedDict):
    """
    Base state that ALL agents must have.
    
    Includes core fields + planning fields required by default graph nodes.
    Agents should EXTEND this with domain-specific fields via inheritance.
    
    Example:
        class ChequeAgentState(BaseAgentState):
            cheque_number: Optional[str]
            micr_data: Optional[Dict]
    """
    
    # === CORE FIELDS ===
    
    # Message history with truncation reducer
    messages: Annotated[List[Dict[str, Any]], message_reducer]
    
    # Session identification
    session_id: str
    
    # Control flow
    should_continue: bool
    error: Optional[str]
    
    # Iteration tracking
    iteration: int
    max_iterations: int
    
    # Model configuration
    model: str
    
    # === PLANNING FIELDS (used by default graph nodes) ===
    # These are required by the default planner/executor/reflector nodes.
    # If you override create_graph() with fully custom nodes, you may
    # omit these from your state (by not inheriting BaseAgentState).
    
    plan: List[str]
    current_step: int
    current_task: Optional[str]
    thoughts: List[str]
    results: Dict[str, Any]
    
    # === OPTIONAL FIELDS (used by process() if documents are passed) ===
    documents: Optional[Dict[str, Any]]


class DocumentInfo(TypedDict):
    """Information about an uploaded document"""
    doc_id: str
    filename: str
    path: str
    content_type: str
    size: int
    uploaded_at: str
    processed: bool


class ProcessingResult(TypedDict, total=False):
    """
    Generic processing result - extend for your domain.
    """
    document_id: str
    status: str  # "success" | "partial" | "failed"
    data: Dict[str, Any]
    confidence: float
    errors: List[str]
    processed_at: str


def create_base_state(session_id: str, model: str = None) -> BaseAgentState:
    """Create initial state with all required fields for default graph nodes."""
    from point9_platform.settings.system import SYSTEM_SETTINGS
    from point9_platform.settings.user import UserSettings
    
    settings = UserSettings()
    
    return BaseAgentState(
        messages=[],
        session_id=session_id,
        should_continue=True,
        error=None,
        iteration=0,
        max_iterations=SYSTEM_SETTINGS.MAX_ITERATIONS,
        model=model or settings.DEFAULT_LLM_MODEL,
        plan=[],
        current_step=0,
        current_task=None,
        thoughts=[],
        results={},
        documents=None,
    )
