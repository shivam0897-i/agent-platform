"""
Validate Data Tool
==================

Example validation tool.
"""

from typing import Dict, Any, List
from point9_platform.tools.decorator import tool


@tool(
    name="validate_data",
    description="Validate extracted data against rules. Use this to check if extracted data is correct and complete."
)
def validate_data(document_id: str, data: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate extracted data.
    
    Args:
        document_id: ID of the document
        data: Data to validate
        state: Current agent state
    
    Returns:
        Validation result
    """
    results = state.get("results", {})
    
    if document_id not in results:
        return {
            "status": "failed",
            "document_id": document_id,
            "error": "No extraction results found for this document"
        }
    
    extraction_result = results[document_id]
    extracted_data = extraction_result.get("data", {})
    
    # Placeholder validation logic
    errors = []
    warnings = []
    
    for field, value in extracted_data.items():
        if not value or value.startswith("<extracted_"):
            warnings.append(f"Field '{field}' may need manual review")
    
    is_valid = len(errors) == 0
    
    return {
        "status": "success" if is_valid else "partial",
        "document_id": document_id,
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "confidence": 0.9 if is_valid else 0.6
    }
