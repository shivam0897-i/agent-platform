"""
Document Agent State
====================

Domain-specific state extending BaseAgentState.
"""

from typing import Dict, Any, List, Optional, Annotated, TypedDict
from point9_platform.agent.state import BaseAgentState, DocumentInfo, ProcessingResult, message_reducer


class DocumentAgentState(TypedDict):
    """
    State for document processing agent.
    
    Extends base state with document-specific fields.
    """
    
    # === BASE FIELDS (required) ===
    messages: Annotated[List[Dict[str, Any]], message_reducer]
    session_id: str
    should_continue: bool
    error: Optional[str]
    iteration: int
    max_iterations: int
    model: str
    
    # === DOCUMENT FIELDS ===
    documents: Dict[str, DocumentInfo]
    results: Dict[str, ProcessingResult]
    
    # === PLANNING FIELDS ===
    plan: List[str]
    current_step: int
    current_task: Optional[str]
    
    # === DEBUG/AUDIT FIELDS ===
    thoughts: List[str]
    needs_human_input: bool
