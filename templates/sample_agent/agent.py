"""
Sample Document Agent
=====================

Reference implementation showing how to create an agent.
"""

from typing import Dict, List, Any
from point9_platform.agent.base import BaseAgent
from sample_agent.state import DocumentAgentState
from sample_agent.settings import DocumentSettings
from sample_agent.prompts.templates import PROMPTS


class DocumentAgent(BaseAgent[DocumentAgentState]):
    """
    Sample document processing agent.
    
    Copy this as a starting point for your own agents.
    """
    
    def __init__(self, session_id: str):
        super().__init__(
            session_id=session_id,
            tools_package="sample_agent.tools",
            settings=DocumentSettings()
        )
    
    def get_agent_name(self) -> str:
        return "document_processor"
    
    def get_domain_keywords(self) -> List[str]:
        return [
            "document", "extract", "process", "data",
            "validate", "compare", "report", "analyze",
            "file", "upload", "scan", "read", "parse"
        ]
    
    def create_initial_state(self, session_id: str) -> DocumentAgentState:
        from point9_platform.settings.system import SYSTEM_SETTINGS
        
        return DocumentAgentState(
            messages=[],
            session_id=session_id,
            documents={},
            results={},
            plan=[],
            current_step=0,
            current_task=None,
            thoughts=[],
            should_continue=True,
            needs_human_input=False,
            error=None,
            iteration=0,
            max_iterations=SYSTEM_SETTINGS.MAX_ITERATIONS,
            model=self.settings.DEFAULT_LLM_MODEL
        )
    
    def get_prompts(self) -> Dict[str, str]:
        return PROMPTS
