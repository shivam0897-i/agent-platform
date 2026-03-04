"""
Document Agent State
====================

Domain-specific state extending BaseAgentState.

NOTE: Use TypedDict inheritance to get all base + planning fields
automatically. Only declare YOUR domain-specific fields here.
"""

from point9_platform.agent.state import BaseAgentState


class DocumentAgentState(BaseAgentState):
    """
    State for document processing agent.
    
    Inherits from BaseAgentState which includes:
    - Core fields: messages, session_id, should_continue, error, iteration, etc.
    - Planning fields: plan, current_step, current_task, thoughts, results, documents
    
    Only add domain-specific fields below.
    """
    
    # === DOMAIN-SPECIFIC FIELDS ===
    needs_human_input: bool
