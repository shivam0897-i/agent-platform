"""
Sample Agent API
================

FastAPI endpoints for the document processing agent.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uuid

from sample_agent.agent import DocumentAgent
from point9_platform.health import create_health_router
from point9_platform.observability.emitter import get_session_emitter

app = FastAPI(
    title="Document Processing Agent",
    description="AI agent for processing and extracting data from documents",
    version="1.0.0"
)

# Include platform health endpoints
app.include_router(create_health_router())


# === Request/Response Models ===

class ProcessRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    documents: Optional[Dict[str, Any]] = None


class ProcessResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: str


# === Endpoints ===

@app.post("/process", response_model=ProcessResponse)
async def process_document(request: ProcessRequest):
    """Process a document with the agent."""
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        agent = DocumentAgent(session_id=session_id)
        result = agent.process(
            message=request.message,
            documents=request.documents
        )
        return ProcessResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ProcessResponse)
async def chat(request: ChatRequest):
    """Continue conversation with the agent."""
    try:
        agent = DocumentAgent(session_id=request.session_id)
        result = agent.process(message=request.message)
        return ProcessResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stream/{session_id}")
async def stream_progress(session_id: str):
    """Stream processing progress via SSE."""
    emitter = get_session_emitter(session_id)
    if not emitter:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return StreamingResponse(
        emitter.stream(),
        media_type="text/event-stream"
    )


# === Run with: uvicorn sample_agent.api.main:app --reload ===
