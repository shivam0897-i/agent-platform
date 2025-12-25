"""
Base Agent State
================

Minimal state schema that all agents must have.
"""

from typing import Dict, Any, List, Optional, Annotated, TypedDict
from datetime import datetime
import operator


def message_reducer(current: List[Dict], new: List[Dict]) -> List[Dict]:
    """
    Custom reducer that truncates message history to prevent token explosion.
    """
    from point9_platform.settings.system import SYSTEM_SETTINGS
    combined = current + new
    return combined[-SYSTEM_SETTINGS.MAX_MESSAGES:]


class BaseAgentState(TypedDict):
    """
    Minimal state that ALL agents must have.
    
    Agents should EXTEND this with domain-specific fields.
    
    Example:
        class ChequeAgentState(BaseAgentState):
            cheque_number: Optional[str]
            micr_data: Optional[Dict]
    """
    
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
    """Create minimal initial state"""
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
        model=model or settings.DEFAULT_LLM_MODEL
    )
