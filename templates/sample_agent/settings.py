"""
Document Agent Settings
=======================

Domain-specific settings extending UserSettings.
"""

from point9_platform.settings.user import UserSettings
from typing import Optional, List


class DocumentSettings(UserSettings):
    """
    Document processing agent settings.
    
    Extends UserSettings with domain-specific configuration.
    """
    
    # === AGENT IDENTITY ===
    AGENT_NAME: str = "Document Processing Agent"
    AGENT_DESCRIPTION: str = "AI agent for processing and extracting data from documents"
    
    # === DOMAIN CONFIGURATION ===
    ALLOWED_OPERATIONS: List[str] = [
        "extract_data",
        "validate_data",
        "compare_documents",
        "generate_report",
    ]
    
    DOMAIN_KEYWORDS: List[str] = [
        "extract", "process", "document", "data", "validate",
        "compare", "report", "analyze", "read", "parse",
    ]
    
    # === EXTERNAL SERVICES (set in .env or config.yaml) ===
    TEXT_EXTRACTION_URL: Optional[str] = None
    
    class Config(UserSettings.Config):
        # Optional: prefix env vars with DOCUMENT_
        # env_prefix = "DOCUMENT_"
        pass
