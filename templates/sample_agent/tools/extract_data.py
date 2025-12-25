"""
Extract Data Tool
=================

Example tool showing how to use @tool decorator.
"""

from typing import Dict, Any, List
from point9_platform.tools.decorator import tool


@tool(
    name="extract_data",
    description="Extract structured data from a document. Use this to pull specific fields from uploaded documents."
)
def extract_data(document_id: str, fields: List[str], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract specified fields from a document.
    
    Args:
        document_id: ID of the document to process
        fields: List of field names to extract
        state: Current agent state (injected by executor)
    
    Returns:
        Extraction result with status and data
    """
    documents = state.get("documents", {})
    
    if document_id not in documents:
        return {
            "status": "failed",
            "document_id": document_id,
            "error": f"Document {document_id} not found"
        }
    
    doc_info = documents[document_id]
    
    # Placeholder extraction logic
    # In real implementation, this would:
    # 1. Load document from path
    # 2. Use LLM/OCR to extract fields
    # 3. Return structured data
    
    extracted_data = {
        field: f"<extracted_{field}>" for field in fields
    }
    
    return {
        "status": "success",
        "document_id": document_id,
        "filename": doc_info.get("filename", "unknown"),
        "data": extracted_data,
        "confidence": 0.85,
        "fields_extracted": len(fields),
        "fields_missing": 0
    }
