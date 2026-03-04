# Point9 Agent Platform

**Version 1.0.0**

A Python SDK for building production-ready AI agents with LangGraph.

---

## What is Point9 Agent Platform?

Point9 Agent Platform is a **reusable foundation** for building AI agents. Instead of writing LangGraph workflows, tool systems, and infrastructure from scratch, you extend a base class and focus only on your domain-specific logic.

### The Problem It Solves

Building an AI agent typically requires:
- Setting up LangGraph with nodes, edges, and state management
- Creating a tool registration and execution system
- Handling LLM provider switching
- Implementing retry logic, error handling, logging
- Building SSE streaming for real-time progress
- Managing session state and conversation memory

**This is 2,000+ lines of boilerplate—repeated for every agent.**

### The Solution

Point9 provides all of this out-of-the-box. You only write:

```python
from point9_platform import BaseAgent, tool

class MyAgent(BaseAgent[MyState]):
    def get_agent_name(self) -> str:
        return "my_agent"
    
    def create_initial_state(self, session_id: str) -> MyState:
        return MyState(session_id=session_id, ...)

@tool(name="my_tool", description="...")
def my_tool(data: str, state: dict) -> dict:
    return {"result": "..."}
```

**~100 lines instead of 2,000+.**

---

## Key Features

| Feature | What It Does |
|---------|--------------|
| **BaseAgent** | Abstract class with full LangGraph workflow (Plan → Execute → Reflect → Respond) |
| **@tool decorator** | Register any function as an agent tool with auto-generated JSON schema |
| **ToolRegistry** | Auto-discovers tools from your `tools/` package |
| **LLMProvider** | Unified interface to OpenAI, Gemini, Claude, Groq, Mistral (via LiteLLM) |
| **S3Storage** | Upload/download files to S3 (images, PDFs, reports) |
| **MongoStore** | Persist session state, logs, chat history to MongoDB |
| **StepEmitter** | Real-time SSE streaming of agent progress to frontend |
| **Colored Logging** | Consistent, noise-filtered console output |
| **Health Endpoints** | Pre-built `/health` and `/ready` routes |

---

## When to Use This Platform

| ✅ Use When | ❌ Don't Use When |
|-------------|-------------------|
| Building multi-step AI workflows | Simple single-prompt LLM calls |
| Need tool/function calling | Just need chat completion |
| Want consistent agent architecture | One-off scripts |
| Multiple agents sharing patterns | Highly custom graph structures |
| Production deployment required | Quick prototypes |

---

## Table of Contents

1. [What is Point9 Agent Platform?](#what-is-point9-agent-platform)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Your First Agent (Tutorial)](#your-first-agent-tutorial)
5. [Platform Architecture](#platform-architecture)
6. [Creating an Agent](#creating-an-agent)
7. [Agent Structure](#agent-structure)
8. [Core Components](#core-components)
9. [Creating Tools](#creating-tools)
10. [Configuration](#configuration)
11. [API Endpoints](#api-endpoints)
12. [Multi-turn Chat](#multi-turn-chat)
13. [Progress Streaming](#progress-streaming)
14. [Deployment](#deployment)
15. [Troubleshooting / FAQ](#troubleshooting--faq)
16. [Examples](#examples)

---

## Installation

### From Private GitHub Repository

**Option 1: Using SSH (if you have SSH keys configured)**
```bash
pip install git+ssh://git@github.com/santhoshmallojwala/unifyriver.git@shivam
```

**Option 2: Using Personal Access Token (PAT)**
```bash
# Replace <YOUR_PAT> with your GitHub Personal Access Token
pip install git+https://<YOUR_PAT>@github.com/santhoshmallojwala/unifyriver.git@shivam
```

**Option 3: In requirements.txt**
```txt
# Using PAT (replace <YOUR_PAT>)
point9-agent-platform @ git+https://<YOUR_PAT>@github.com/santhoshmallojwala/unifyriver.git@shivam

# Or using SSH
point9-agent-platform @ git+ssh://git@github.com/santhoshmallojwala/unifyriver.git@shivam
```

### For Development (Editable Install)

```bash
git clone https://github.com/santhoshmallojwala/unifyriver.git
cd agent-platform
pip install -e .
```

### With Optional Dependencies

```bash
# With storage utilities (S3 + MongoDB)
pip install "point9-agent-platform[storage] @ git+https://..."

# With development tools
pip install "point9-agent-platform[dev] @ git+https://..."
```

### What Gets Installed

| Package | Purpose |
|---------|---------|
| `langgraph` | Workflow orchestration |
| `litellm` | Multi-provider LLM interface |
| `fastapi` | API framework |
| `pydantic` | Data validation |
| `pyyaml` | Config file parsing |
| `boto3` *(optional)* | S3 storage |
| `pymongo` *(optional)* | MongoDB storage |

### Verify Installation

```python
from point9_platform import BaseAgent, tool, ToolRegistry, UserSettings, SYSTEM_SETTINGS
print(f"Platform v{SYSTEM_SETTINGS.PLATFORM_VERSION}")  # Platform v1.0.0
```

---

## Quick Start

### Step 1: Copy the Template

```bash
cp -r templates/sample_agent ../invoice_agent
cd ../invoice_agent
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure

Edit `.env`:
```env
GEMINI_API_KEY=your-api-key
```

Edit `config.yaml`:
```yaml
AGENT_NAME: Invoice Processing Agent
```

### Step 4: Run

```bash
uvicorn api.main:app --reload --port 8000
```

### Step 5: Test

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"message": "Process this invoice"}'
```

---

## Your First Agent (Tutorial)

This tutorial walks you through creating a complete agent from scratch. By the end, you'll have a working **Document Summarizer Agent**.

### Step 1: Create Project Structure

```bash
mkdir summarizer_agent
cd summarizer_agent
mkdir -p api tools prompts
touch agent.py state.py settings.py requirements.txt .env
touch api/__init__.py api/main.py
touch tools/__init__.py tools/summarize.py
touch prompts/__init__.py prompts/templates.py
```

Your folder structure:
```
summarizer_agent/
├── agent.py           # Main agent class
├── state.py           # Agent state definition
├── settings.py        # Custom settings
├── requirements.txt   # Dependencies
├── .env               # API keys
├── api/
│   ├── __init__.py
│   └── main.py        # FastAPI endpoints
├── tools/
│   ├── __init__.py
│   └── summarize.py   # Your tools
└── prompts/
    ├── __init__.py
    └── templates.py   # LLM prompts
```

### Step 2: Create requirements.txt

```txt
# Point9 Platform (from private repo)
point9-agent-platform @ git+https://github.com/santhoshmallojwala/unifyriver.git@shivam

# Web framework
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6

# Environment management
python-dotenv>=1.0.0
```

### Step 3: Create .env

```env
GEMINI_API_KEY=your-gemini-api-key-here
```

### Step 4: Create state.py

```python
"""Agent State Definition"""
from typing import Dict, Any, List, Optional, Annotated, TypedDict
from point9_platform.agent.state import message_reducer, DocumentInfo


class SummarizerState(TypedDict):
    """State for the Summarizer Agent."""
    
    # === Required fields (from platform) ===
    messages: Annotated[List[Dict[str, Any]], message_reducer]
    session_id: str
    should_continue: bool
    error: Optional[str]
    iteration: int
    max_iterations: int
    model: str
    
    # === Standard fields ===
    documents: Dict[str, DocumentInfo]
    results: Dict[str, Any]
    plan: List[str]
    current_step: int
    thoughts: List[str]
    
    # === Custom fields for this agent ===
    summary: Optional[str]
    word_count: int
    key_points: List[str]
```

### Step 5: Create prompts/templates.py

```python
"""LLM Prompts for Summarizer Agent"""

PLANNER_PROMPT = """You are a document summarization planner.

Given a user request, create a plan to summarize the document.

Respond in JSON:
{
    "task_understanding": "What the user wants",
    "reasoning": "How you'll approach this",
    "plan": ["Step 1", "Step 2", ...]
}
"""

RESPONDER_PROMPT = """You are a helpful assistant that summarizes documents.

Based on the summarization results, provide a clear, helpful response to the user.
Include the summary and key points in your response.
"""

PROMPTS = {
    "planner": PLANNER_PROMPT,
    "responder": RESPONDER_PROMPT
}
```

### Step 6: Create agent.py

```python
"""Summarizer Agent - Main Agent Class"""
from typing import Dict, List, Any
from point9_platform.agent.base import BaseAgent
from state import SummarizerState
from settings import SummarizerSettings
from prompts.templates import PROMPTS


class SummarizerAgent(BaseAgent[SummarizerState]):
    """AI agent for summarizing documents."""
    
    def __init__(self, session_id: str):
        super().__init__(
            session_id=session_id,
            tools_package="tools",  # Auto-discovers tools from tools/ folder
            settings=SummarizerSettings()
        )
    
    # === REQUIRED METHODS ===
    
    def get_agent_name(self) -> str:
        return "summarizer_agent"
    
    def get_domain_keywords(self) -> List[str]:
        return ["summarize", "summary", "document", "text", "analyze", "brief"]
    
    def create_initial_state(self, session_id: str) -> SummarizerState:
        return SummarizerState(
            messages=[],
            session_id=session_id,
            should_continue=True,
            error=None,
            iteration=0,
            max_iterations=10,
            model=self.settings.DEFAULT_LLM_MODEL,
            documents={},
            results={},
            plan=[],
            current_step=0,
            thoughts=[],
            summary=None,
            word_count=0,
            key_points=[]
        )
    
    def get_prompts(self) -> Dict[str, str]:
        return PROMPTS
    
    # === OPTIONAL: Override state class for TypedDict support ===
    
    def get_state_class(self):
        return SummarizerState
```

### Step 7: Create settings.py

```python
"""Custom Settings for Summarizer Agent"""
from point9_platform.settings.user import UserSettings


class SummarizerSettings(UserSettings):
    """Settings for the Summarizer Agent."""
    
    # Agent identity
    AGENT_NAME: str = "Document Summarizer"
    AGENT_DESCRIPTION: str = "AI agent for summarizing documents and extracting key points"
    
    # LLM settings
    DEFAULT_LLM_MODEL: str = "gemini/gemini-2.0-flash"
    LLM_TEMPERATURE: float = 0.3
    
    # Custom settings
    MAX_SUMMARY_LENGTH: int = 500
    EXTRACT_KEY_POINTS: bool = True
```

### Step 8: Create tools/summarize.py

```python
"""Summarization Tool"""
from point9_platform.tools.decorator import tool
from typing import Dict, Any, List


@tool(
    name="summarize_text",
    description="Summarize the given text and extract key points"
)
def summarize_text(
    text: str,
    max_length: int = 500,
    extract_points: bool = True,
    state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Summarize text content.
    
    Args:
        text: The text to summarize
        max_length: Maximum length of summary in words
        extract_points: Whether to extract key bullet points
        state: Agent state (automatically injected)
    
    Returns:
        Dict with summary, word_count, and key_points
    """
    # Simple summarization logic (replace with your own)
    words = text.split()
    word_count = len(words)
    
    # Create summary (first N words as demo)
    summary_words = words[:min(max_length, len(words))]
    summary = " ".join(summary_words)
    if len(words) > max_length:
        summary += "..."
    
    # Extract key points (demo: first 3 sentences)
    key_points = []
    if extract_points:
        sentences = text.split(". ")
        key_points = [s.strip() + "." for s in sentences[:3] if s.strip()]
    
    return {
        "status": "success",
        "summary": summary,
        "word_count": word_count,
        "key_points": key_points,
        "original_length": len(text),
        "summary_length": len(summary)
    }


@tool(
    name="analyze_document",
    description="Analyze a document and provide statistics"
)
def analyze_document(
    document_id: str,
    state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Analyze an uploaded document.
    
    Args:
        document_id: ID of the uploaded document
        state: Agent state (automatically injected)
    
    Returns:
        Dict with document statistics
    """
    documents = state.get("documents", {})
    doc = documents.get(document_id)
    
    if not doc:
        return {
            "status": "failed",
            "error": f"Document {document_id} not found"
        }
    
    # Get document info
    filename = doc.get("filename", "unknown")
    file_type = doc.get("type", "unknown")
    content = doc.get("content", "")
    
    return {
        "status": "success",
        "document_id": document_id,
        "filename": filename,
        "file_type": file_type,
        "character_count": len(content),
        "word_count": len(content.split()),
        "line_count": len(content.split("\n"))
    }
```

### Step 9: Create api/main.py

```python
"""FastAPI Application"""
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional

from point9_platform.health import create_health_router
from point9_platform.observability.emitter import get_session_emitter

# Import your agent
import sys
sys.path.insert(0, "..")
from agent import SummarizerAgent


app = FastAPI(
    title="Summarizer Agent API",
    description="AI agent for document summarization",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Health endpoints
app.include_router(create_health_router())


# Request/Response models
class ProcessRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    documents: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    message: str
    session_id: str


# Endpoints
@app.post("/process")
async def process(request: ProcessRequest):
    """Process a summarization request."""
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        agent = SummarizerAgent(session_id=session_id)
        result = agent.process(
            message=request.message,
            documents=request.documents
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    """Continue a conversation."""
    try:
        agent = SummarizerAgent(session_id=request.session_id)
        result = agent.process(message=request.message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stream/{session_id}")
async def stream(session_id: str):
    """Stream progress updates via SSE."""
    emitter = get_session_emitter(session_id)
    if not emitter:
        raise HTTPException(status_code=404, detail="Session not found")
    return StreamingResponse(
        emitter.stream(),
        media_type="text/event-stream"
    )
```

### Step 10: Run Your Agent

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
cd api
uvicorn main:app --reload --port 8000
```

### Step 11: Test Your Agent

```bash
# Health check
curl http://localhost:8000/health

# Process a request
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Please summarize this text: Artificial intelligence is transforming industries worldwide. From healthcare to finance, AI is making processes more efficient and accurate. Machine learning models can now analyze vast amounts of data in seconds."
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Here's your summary: AI is transforming industries...",
  "results": {
    "summarize_text": {
      "status": "success",
      "summary": "...",
      "word_count": 35,
      "key_points": ["..."]
    }
  },
  "session_id": "abc-123"
}
```

---

## Platform Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     YOUR CUSTOM AGENT                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
│  │ agent.py│  │ tools/  │  │prompts/ │  │ domain folders/ │ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────────┬────────┘ │
│       │            │            │                │          │
└───────┼────────────┼────────────┼────────────────┼──────────┘
        │            │            │                │
        ▼            ▼            ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                   POINT9 PLATFORM                            │
│  ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ BaseAgent │ │ @tool    │ │ Settings │ │ LLMProvider   │  │
│  │ (ABC)     │ │ decorator│ │ (YAML)   │ │ (LiteLLM)     │  │
│  └───────────┘ └──────────┘ └──────────┘ └───────────────┘  │
│  ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ Graph     │ │ Tool     │ │ SSE      │ │ Health        │  │
│  │ Builder   │ │ Registry │ │ Emitter  │ │ Endpoints     │  │
│  └───────────┘ └──────────┘ └──────────┘ └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Agent Processing Flow

When a request comes in, here's exactly what happens:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AGENT PROCESSING FLOW                            │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │ API Request  │  POST /process {"message": "...", "documents": {...}}
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │agent.process │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐     ┌─────────────────┐
    │validate_domain│────▶│ Invalid? Return │
    └──────┬───────┘     │ error response  │
           │ Valid       └─────────────────┘
           ▼
    ┌───────────────────┐
    │create_initial_state│
    └──────┬────────────┘
           │
           ▼
    ┌──────────────┐
    │  on_start()  │  ──▶ Emit SSE: AGENT_PLANNING
    └──────┬───────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│                    LANGGRAPH WORKFLOW                         │
│                                                               │
│   ┌─────────────┐                                             │
│   │Planner Node │ ──▶ LLM creates execution plan              │
│   └──────┬──────┘                                             │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────┐     ┌────────────────────┐                  │
│   │Executor Node│────▶│ToolExecutor.execute│                  │
│   └──────┬──────┘     │  • Emit: TOOL_CALL │                  │
│          │            │  • Run tool_fn()   │                  │
│          │            │  • Retry if failed │                  │
│          │            │  • Emit: TOOL_RESULT│                 │
│          │            └────────────────────┘                  │
│          │                     │                              │
│          │◀────────────────────┘ (loop for more tools)        │
│          │                                                    │
│          ▼                                                    │
│   ┌──────────────┐                                            │
│   │Reflector Node│ ──▶ LLM reviews: continue or done?         │
│   └──────┬───────┘                                            │
│          │                                                    │
│          │ Continue? ───▶ Back to Executor                    │
│          │                                                    │
│          ▼ Done                                               │
│   ┌──────────────┐                                            │
│   │Responder Node│ ──▶ LLM generates final response           │
│   └──────────────┘                                            │
│                                                               │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────┐
                    │ _extract_result() │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌──────────────┐
                    │ on_complete()│ ──▶ Emit SSE: COMPLETE
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │Return response│ {"success": true, "message": "..."}
                    └──────────────┘
```

### Detailed Step-by-Step Flow

```
1. API LAYER
   └── POST /process {"message": "...", "documents": {...}}
       └── agent = MyAgent(session_id)
           └── agent.process(message, documents)

2. VALIDATION
   └── validate_domain(message)
       └── Check if message contains domain keywords
       └── If no match → return {"success": false, "error": "..."}

3. STATE INITIALIZATION
   └── state = create_initial_state(session_id)
       └── messages: [{"role": "user", "content": message}]
       └── documents: {...}
       └── results: {}
       └── iteration: 0

4. LIFECYCLE HOOK
   └── on_start(state)
       └── Emit SSE: {"step_type": "agent_planning", ...}

5. LANGGRAPH EXECUTION
   └── graph.invoke(state, config)
   
   5a. PLANNER NODE
       └── LLM call with planner prompt
       └── Returns: plan = ["Step 1", "Step 2", ...]
   
   5b. EXECUTOR NODE (loop)
       └── LLM selects tool from available tools
       └── ToolExecutor.execute(tool_name, args)
           └── Emit SSE: {"step_type": "tool_call", ...}
           └── tool_fn(**args, state=state)
           └── Retry on failure (3 attempts, exponential backoff)
           └── Emit SSE: {"step_type": "tool_result", ...}
   
   5c. REFLECTOR NODE
       └── LLM reviews results
       └── Decides: continue or respond
   
   5d. RESPONDER NODE
       └── LLM generates final response
       └── Appends to state["messages"]

6. RESULT EXTRACTION
   └── _extract_result(final_state)
       └── Get last assistant message
       └── Get results from state

7. LIFECYCLE HOOK
   └── on_complete(state, result)
       └── Emit SSE: {"step_type": "complete", "progress": 100}

8. RETURN
   └── {"success": true, "message": "...", "results": {...}}
```

---

## Creating an Agent

### Minimal Agent (3 Files)

```
simple_agent/
├── agent.py          # Required
├── tools/
│   └── my_tool.py    # At least one tool
└── requirements.txt
```

### Standard Agent (Template)

```
my_agent/
├── agent.py
├── state.py
├── settings.py
├── config.yaml
├── .env
├── requirements.txt
├── api/main.py
├── prompts/templates.py
└── tools/
    ├── tool_a.py
    └── tool_b.py
```

### Complex Agent (Multiple Domain Folders)

For complex agents like LC Validation or Document Processing:

```
lc_validation_agent/
├── agent.py                  # Main agent
├── state.py                  # Agent state
├── settings.py               # Custom settings
├── config.yaml
├── .env
├── requirements.txt
│
├── api/                      # FastAPI endpoints
│   ├── __init__.py
│   └── main.py
│
├── prompts/                  # LLM prompts
│   ├── __init__.py
│   └── templates.py
│
├── tools/                    # Auto-discovered tools
│   ├── __init__.py
│   ├── extract_lc.py
│   ├── validate_ucp.py
│   └── compare_docs.py
│
├── extraction/               # Domain: Data extraction
│   ├── __init__.py
│   ├── field_extractor.py
│   ├── table_parser.py
│   └── ocr_client.py
│
├── validation/               # Domain: Validation rules
│   ├── __init__.py
│   ├── ucp_rules.py
│   ├── compliance_checker.py
│   └── discrepancy_detector.py
│
├── integrations/             # Domain: External services
│   ├── __init__.py
│   ├── swift_client.py
│   ├── bank_api.py
│   └── document_store.py
│
├── rag/                      # Domain: RAG/Knowledge
│   ├── __init__.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── retriever.py
│
└── utils/                    # Shared utilities
    ├── __init__.py
    ├── formatters.py
    └── validators.py
```

**You can add ANY folders your domain needs!** The platform doesn't limit structure.

---

## Agent Structure

### agent.py — Main Agent Class

```python
from typing import Dict, List, Any
from point9_platform.agent.base import BaseAgent
from my_agent.state import MyAgentState
from my_agent.settings import MySettings
from my_agent.prompts.templates import PROMPTS


class MyAgent(BaseAgent[MyAgentState]):
    """My custom agent."""
    
    def __init__(self, session_id: str):
        super().__init__(
            session_id=session_id,
            tools_package="my_agent.tools",  # Where tools are discovered
            settings=MySettings()
        )
    
    # === REQUIRED: Must implement these ===
    
    def get_agent_name(self) -> str:
        return "my_agent"
    
    def get_domain_keywords(self) -> List[str]:
        return ["invoice", "bill", "payment", "vendor"]
    
    def create_initial_state(self, session_id: str) -> MyAgentState:
        return MyAgentState(
            messages=[],
            session_id=session_id,
            documents={},
            results={},
            # ... your custom fields
        )
    
    def get_prompts(self) -> Dict[str, str]:
        return PROMPTS
    
    # === OPTIONAL: Override for custom behavior ===
    
    def create_graph(self):
        # Override for completely custom workflow
        pass
    
    def on_start(self, state):
        # Custom setup before processing
        pass
    
    def on_complete(self, state, result):
        # Custom cleanup after processing
        pass
```

### state.py — Agent State

```python
from typing import Dict, Any, List, Optional, Annotated, TypedDict
from point9_platform.agent.state import message_reducer, DocumentInfo


class MyAgentState(TypedDict):
    # === Base fields (required) ===
    messages: Annotated[List[Dict[str, Any]], message_reducer]
    session_id: str
    should_continue: bool
    error: Optional[str]
    iteration: int
    max_iterations: int
    model: str
    
    # === Your custom fields ===
    documents: Dict[str, DocumentInfo]
    results: Dict[str, Any]
    
    # Add domain-specific fields:
    invoice_number: Optional[str]
    vendor_name: Optional[str]
    line_items: List[Dict]
    validation_status: str
```

### settings.py — Custom Settings

```python
from point9_platform.settings.user import UserSettings
from typing import Optional, List


class MySettings(UserSettings):
    """Custom settings for my agent."""
    
    # Override defaults
    AGENT_NAME: str = "Invoice Agent"
    AGENT_DESCRIPTION: str = "AI agent for processing invoices"
    
    # Add custom settings
    CONFIDENCE_THRESHOLD: float = 0.85
    MAX_LINE_ITEMS: int = 100
    VALIDATION_STRICT: bool = True
    
    # External services
    OCR_SERVICE_URL: Optional[str] = None
    ERP_API_URL: Optional[str] = None
    
    ALLOWED_DOCUMENT_TYPES: List[str] = ["pdf", "png", "jpg"]
```

---

## Core Components

### BaseAgent

Abstract base class that all agents extend.

| Method | Required | Purpose |
|--------|----------|---------|
| `get_agent_name()` | ✅ Yes | Return unique agent identifier |
| `get_domain_keywords()` | ✅ Yes | Keywords for domain validation |
| `create_initial_state()` | ✅ Yes | Create initial state for session |
| `get_prompts()` | ✅ Yes | Return prompt templates |
| `create_graph()` | No | Override for custom workflow |
| `create_planner_node()` | No | Override for custom planning |
| `create_executor_node()` | No | Override for custom execution |
| `on_start()` | No | Lifecycle hook before processing |
| `on_complete()` | No | Lifecycle hook after success |
| `on_error()` | No | Lifecycle hook on failure |

### @tool Decorator

Register functions as agent tools.

```python
from point9_platform.tools.decorator import tool
from typing import Dict, Any, List


@tool(
    name="extract_invoice_data",
    description="Extract structured data from an invoice document"
)
def extract_invoice_data(
    document_id: str,
    fields: List[str],
    state: Dict[str, Any]  # Always injected
) -> Dict[str, Any]:
    """
    Extract invoice fields.
    
    Args:
        document_id: ID of uploaded document
        fields: Fields to extract ["vendor", "amount", "date"]
        state: Current agent state (injected by executor)
    
    Returns:
        Extraction result with status and data
    """
    # Access documents from state
    documents = state.get("documents", {})
    doc = documents.get(document_id)
    
    if not doc:
        return {"status": "failed", "error": "Document not found"}
    
    # Your extraction logic here
    extracted = {
        "vendor": "ACME Corp",
        "amount": 1500.00,
        "date": "2024-01-15"
    }
    
    return {
        "status": "success",
        "document_id": document_id,
        "data": extracted,
        "confidence": 0.92
    }
```

**Features:**
- Auto-generates JSON schema from type hints
- `state` parameter is automatically injected
- Retry with exponential backoff on failure
- Progress tracking via SSE

### UserSettings

Configuration with multiple sources.

**Priority (highest to lowest):**
1. Environment variables
2. `.env` file
3. `config.yaml` file
4. Default values in class

```python
from point9_platform.settings.user import UserSettings

class MySettings(UserSettings):
    MY_SETTING: str = "default"  # Can be overridden
```

### SYSTEM_SETTINGS

Immutable platform constants (do NOT override).

```python
from point9_platform.settings.system import SYSTEM_SETTINGS

SYSTEM_SETTINGS.MAX_ITERATIONS      # 10
SYSTEM_SETTINGS.TOOL_RETRY_ATTEMPTS # 3
SYSTEM_SETTINGS.LLM_TIMEOUT         # 60
SYSTEM_SETTINGS.PLATFORM_VERSION    # "1.0.0"
```

### LLMProvider

Multi-provider LLM abstraction.

```python
from point9_platform.llm.provider import get_llm_provider

llm = get_llm_provider()

# Simple completion
response = llm.completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="gemini/gemini-2.0-flash"
)

# With tools
response = llm.completion(
    messages=messages,
    tools=tool_definitions,
    tool_choice="auto"
)

# With fallback
response = llm.completion(
    messages=messages,
    model="gemini/gemini-2.0-flash",
    fallback_model="openai/gpt-4o-mini"  # If primary fails
)
```

**Supported Models:**
- Gemini: `gemini/gemini-2.0-flash`, `gemini/gemini-1.5-pro`
- OpenAI: `openai/gpt-4o`, `openai/gpt-4o-mini`
- Anthropic: `anthropic/claude-3.5-sonnet`
- Groq: `groq/llama-3.1-70b`
- Mistral: `mistral/mistral-large`

### ToolRegistry

Auto-discovers tools from a package.

```python
from point9_platform.tools.registry import ToolRegistry

registry = ToolRegistry("my_agent.tools")
registry.discover()

# Get tool definitions for LLM
definitions = registry.get_definitions()

# Get tool function
func = registry.get_function("extract_invoice_data")

# List all tools
names = registry.get_all_names()  # ["extract_invoice_data", "validate_data"]
```

### Storage (S3 & MongoDB)

Reusable storage utilities for files and session state.

**Install storage dependencies:**
```bash
pip install point9-agent-platform[storage]
# or: pip install boto3 pymongo
```

**S3Storage — File Storage:**
```python
from point9_platform.storage import S3Storage, get_s3_storage

s3 = get_s3_storage()

# Upload
s3.upload_file("local.pdf", "inputs/session-123/doc.pdf")
s3.upload_json({"results": data}, "outputs/session-123/results.json")

# Download
s3.download_file("inputs/doc.pdf", "/tmp/doc.pdf")

# Presigned URL (for frontend)
url = s3.get_presigned_url("outputs/report.pdf", expiration=3600)
```

**MongoStore — Session State:**
```python
from point9_platform.storage import MongoStore, get_mongo_store

store = get_mongo_store()

# Session management
store.create_session("session-123", input_files=[...])
store.update_status("session-123", "processing")
store.set_output("session-123", "s3://bucket/output.json")

# Logging
store.add_log("session-123", "extraction", "Extracted 5 fields")

# Intermediate results
store.store_result("session-123", "analyze_image", {"defects": [...]})

# Chat history
store.add_message("session-123", "user", "Process this document")
history = store.get_chat_history("session-123")
```

**Environment Variables:**
```env
# S3
S3_BUCKET_NAME=my-bucket
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=point9_agent
MONGODB_COLLECTION=sessions
```

---

### 🔬 Evaluation Framework

Built-in evaluation for any Point9 agent — RAGAS, HuggingFace Evaluate, and LLM-as-Judge.

```python
from point9_platform.evaluation import Evaluator, get_evaluator

# Using singleton (runs all available scorers)
evaluator = get_evaluator()
result = evaluator.evaluate(
    query="What is the refund policy?",
    context=["Our policy allows returns within 30 days."],
    response="You can return items within 30 days.",
    reference="Refund policy: 30-day return window.",  # optional
)

print(result.to_dict())
# {
#   "ragas":     {"context_precision": 0.92, "faithfulness": 0.95, ...},
#   "hf":        {"rouge1": 0.87, "rougeL": 0.82, "bertscore_f1": 0.91, ...},
#   "llm_judge": {"hallucination_score": 0.05, "self_consistency": 0.93, "content_safety_score": 0.0},
# }

# Cherry-pick scorers
evaluator = Evaluator(run_ragas=True, run_hf=False, run_llm_judge=True)

# Use individual scorers directly
from point9_platform.evaluation import RagasScorer, HFScorer, LLMJudge
judge = LLMJudge(model="groq/llama-3.3-70b-versatile")
scores = judge.score(query=..., context=[...], response=...)

# Async support for FastAPI endpoints
result = await evaluator.aevaluate(query=..., context=[...], response=...)
```

Install evaluation dependencies:
```bash
pip install "point9-agent-platform[eval]"
```

| Scorer | Metrics | Requires |
|--------|---------|----------|
| `RagasScorer` | Context Precision, Context Recall, Faithfulness, Answer Relevance | `ragas`, `datasets` |
| `HFScorer` | ROUGE-1, ROUGE-L, BERTScore (P/R/F1) | `evaluate`, `rouge_score`, `bert_score` |
| `LLMJudge` | Hallucination Score, Self-Consistency, Content Safety | `litellm` |

---

### Logging

Consistent, color-coded logging for all agents.

**Setup (in your main.py or app startup):**
```python
from point9_platform.observability import setup_logging, get_logger

# Configure at app start
setup_logging(
    level="INFO",           # DEBUG, INFO, WARNING, ERROR
    agent_name="my_agent",  # Prefix for all loggers
    filter_noise=True       # Suppress noisy third-party logs
)
```

**Usage (in any module):**
```python
from point9_platform.observability import get_logger

logger = get_logger("tools")
logger.info("Processing document")
logger.warning("Missing field")
logger.error("Validation failed")
```

**Output:**
```
17:54:44 │ INFO  │ tools           │ Processing document
17:54:44 │ WARN  │ tools           │ Missing field
17:54:44 │ ERROR │ tools           │ Validation failed
```

---

## Creating Tools

### Step 1: Create File

Create `tools/my_tool.py`:

```python
from point9_platform.tools.decorator import tool
from typing import Dict, Any


@tool(name="my_tool", description="What this tool does")
def my_tool(param1: str, param2: int, state: Dict[str, Any]) -> Dict[str, Any]:
    # Your logic
    return {"status": "success", "result": "..."}
```

### Step 2: Done!

The tool is auto-discovered. No registration needed.

### Tool Best Practices

```python
@tool(name="process_document", description="Process uploaded document")
def process_document(
    document_id: str,
    options: Dict[str, Any],
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Always include:
    1. Clear docstring
    2. Type hints for all parameters
    3. state parameter (injected)
    4. Return dict with 'status' key
    """
    
    # Access state data
    documents = state.get("documents", {})
    settings = state.get("settings", {})
    
    # Validate inputs
    if document_id not in documents:
        return {
            "status": "failed",
            "document_id": document_id,
            "error": "Document not found"
        }
    
    # Your logic
    result = do_processing(documents[document_id])
    
    # Return structured result
    return {
        "status": "success",
        "document_id": document_id,
        "data": result,
        "confidence": 0.95
    }
```

### Using Domain Code in Tools

Tools can import from your domain folders:

```python
# tools/extract_lc.py
from point9_platform.tools.decorator import tool
from my_agent.extraction.field_extractor import FieldExtractor
from my_agent.validation.ucp_rules import UCPValidator


@tool(name="extract_lc_fields", description="Extract LC fields")
def extract_lc_fields(document_id: str, state: dict) -> dict:
    # Use domain code
    extractor = FieldExtractor()
    validator = UCPValidator()
    
    doc = state["documents"][document_id]
    fields = extractor.extract(doc)
    validated = validator.validate(fields)
    
    return {"status": "success", "data": validated}
```

---

## Configuration

### config.yaml

Non-sensitive configuration:

```yaml
# Agent identity
AGENT_NAME: LC Validation Agent
AGENT_DESCRIPTION: AI agent for validating Letters of Credit

# LLM settings
DEFAULT_LLM_MODEL: gemini/gemini-2.0-flash
LLM_TEMPERATURE: 0.1
LLM_MAX_TOKENS: 4096

# Custom settings
CONFIDENCE_THRESHOLD: 0.85
VALIDATION_MODE: strict
MAX_DOCUMENTS: 10
```

### .env

Secrets (never commit to git):

```env
# LLM API Keys
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key

# External services
OCR_SERVICE_URL=https://your-ocr-service.com/api
DATABASE_URL=mongodb://localhost:27017/mydb
```

### Accessing Settings

```python
from my_agent.settings import MySettings

settings = MySettings()
print(settings.AGENT_NAME)
print(settings.CONFIDENCE_THRESHOLD)
print(settings.GEMINI_API_KEY)  # From .env
```

---

## API Endpoints

### Standard Endpoints

```python
# api/main.py
from fastapi import FastAPI
from point9_platform.health import create_health_router

app = FastAPI(title="My Agent API")
app.include_router(create_health_router())

@app.post("/process")
async def process(request: ProcessRequest):
    agent = MyAgent(session_id=request.session_id)
    return agent.process(request.message, request.documents)

@app.post("/chat")
async def chat(request: ChatRequest):
    agent = MyAgent(session_id=request.session_id)
    return agent.process(request.message)

@app.get("/stream/{session_id}")
async def stream(session_id: str):
    emitter = get_session_emitter(session_id)
    return StreamingResponse(emitter.stream(), media_type="text/event-stream")
```

### Endpoint Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (from platform) |
| `/ready` | GET | Readiness with dependency checks |
| `/process` | POST | Main processing with documents |
| `/chat` | POST | Continue conversation |
| `/stream/{session_id}` | GET | SSE progress stream |

---

## Multi-turn Chat

The platform maintains conversation memory:

```python
# Request 1
POST /process
{"session_id": "user-123", "message": "Process this invoice"}

# Response 1
{"message": "Invoice processed. Total: $1,500", "session_id": "user-123"}

# Request 2 (same session_id)
POST /chat
{"session_id": "user-123", "message": "What was the vendor name?"}

# Response 2 (agent remembers context!)
{"message": "The vendor is ACME Corp", "session_id": "user-123"}
```

---

## Progress Streaming

Send real-time updates to frontend via SSE:

```python
from point9_platform.observability.emitter import (
    get_session_emitter, StepType, StepStatus
)

emitter = get_session_emitter(session_id)

# Emit progress
emitter.emit_blocking(
    StepType.AGENT_PLANNING,
    "Analyzing document structure...",
    status=StepStatus.IN_PROGRESS,
    progress=10
)

emitter.emit_blocking(
    StepType.TOOL_CALL,
    "Extracting invoice fields...",
    status=StepStatus.IN_PROGRESS,
    progress=40
)

emitter.complete_blocking(
    success=True,
    final_message="Processing complete"
)
```

**Frontend receives:**
```
data: {"step_type": "agent_planning", "message": "Analyzing...", "progress": 10}
data: {"step_type": "tool_call", "message": "Extracting...", "progress": 40}
data: {"step_type": "complete", "message": "Processing complete", "progress": 100}
```

---

## Deployment

### Hugging Face Spaces (With Private Repo)

Since Point9 Platform is a private repository, you need to use a **GitHub Personal Access Token (PAT)** to install it on Hugging Face Spaces.

**Step 1: Create a GitHub PAT**
1. Go to GitHub → Settings → Developer Settings → Personal Access Tokens → Fine-grained tokens
2. Create a new token with `read` access to the `agent-platform` repository
3. Copy the token

**Step 2: Add PAT as Hugging Face Secret**
1. Go to your Space → Settings → Repository secrets
2. Add a secret named `GITHUB_PAT` with your token value

**Step 3: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Build argument for GitHub PAT
ARG GITHUB_PAT

# Install dependencies (using PAT for private repo)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Step 4: Create requirements.txt with PAT placeholder**

```txt
# The ${GITHUB_PAT} will be replaced during Docker build
point9-agent-platform @ git+https://${GITHUB_PAT}@github.com/santhoshmallojwala/unifyriver.git@shivam

fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
python-dotenv>=1.0.0
```

**Step 5: Create Hugging Face Space settings**

In your Space, create a `README.md` with frontmatter:

```yaml
---
title: My Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
```

**Step 6: Push and deploy**

The Space will automatically build using the Dockerfile and inject the `GITHUB_PAT` secret.

### Docker (Local/Self-Hosted)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install git for private repo access
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Build and run:**
```bash
# Build (pass PAT as build arg)
docker build --build-arg GITHUB_PAT=your_pat_here -t my-agent .

# Run
docker run -p 7860:7860 --env-file .env my-agent
```

### Cloud Platforms

**AWS ECS / GCP Cloud Run / Azure Container Apps:**
1. Build Docker image locally or in CI/CD
2. Push to container registry (ECR, GCR, ACR)
3. Deploy as container service
4. Inject environment variables from secrets manager

---

## Troubleshooting / FAQ

### Installation Issues

**Q: "ModuleNotFoundError: No module named 'point9_platform'"**

The platform isn't installed. Install it:
```bash
pip install git+https://github.com/santhoshmallojwala/unifyriver.git@shivam
```

If using a private repo, use PAT:
```bash
pip install git+https://<YOUR_PAT>@github.com/santhoshmallojwala/unifyriver.git@shivam
```

---

**Q: "Permission denied" when installing from GitHub**

Your PAT doesn't have access or has expired. Generate a new one:
1. GitHub → Settings → Developer Settings → Personal Access Tokens
2. Create token with `repo` read access
3. Use in pip install command

---

**Q: "ERROR: Could not find a version that satisfies the requirement"**

Check your requirements.txt format:
```txt
# Correct format
point9-agent-platform @ git+https://github.com/santhoshmallojwala/unifyriver.git@shivam

# Wrong format
point9-agent-platform==1.0.0  # Not on PyPI!
```

---

### Agent Issues

**Q: "Request outside [agent_name] domain"**

Your message doesn't match domain keywords. Check `get_domain_keywords()`:
```python
def get_domain_keywords(self) -> List[str]:
    return ["summarize", "document", "text"]  # Message must contain one of these
```

Solution: Add more keywords or modify the request.

---

**Q: "Tool not discovered" or "No tools found"**

1. Check your `tools_package` path in `__init__`:
```python
super().__init__(
    session_id=session_id,
    tools_package="my_agent.tools",  # Must match your package structure
)
```

2. Ensure tools have the `@tool` decorator:
```python
from point9_platform.tools.decorator import tool

@tool(name="my_tool", description="...")
def my_tool(...):
    ...
```

3. Check `tools/__init__.py` exists (even if empty)

---

**Q: "LLM returned invalid response" or "list index out of range"**

The LLM returned an empty response. Check:
1. API key is set in `.env`
2. Model name is correct (e.g., `gemini/gemini-2.0-flash`)
3. You have API quota remaining

---

**Q: "State field not found" or TypedDict errors**

Your state TypedDict is missing required fields. Ensure you have all base fields:
```python
class MyState(TypedDict):
    # Required base fields
    messages: Annotated[List[Dict[str, Any]], message_reducer]
    session_id: str
    should_continue: bool
    error: Optional[str]
    iteration: int
    max_iterations: int
    model: str
    
    # Standard fields
    documents: Dict[str, DocumentInfo]
    results: Dict[str, Any]
    plan: List[str]
    current_step: int
    thoughts: List[str]
```

---

### Configuration Issues

**Q: "GEMINI_API_KEY not found"**

1. Create `.env` file in your agent root:
```env
GEMINI_API_KEY=your-key-here
```

2. Load it in your app:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

**Q: How do I switch LLM providers?**

Set the model in your state or settings:
```python
# In settings.py
DEFAULT_LLM_MODEL: str = "openai/gpt-4o-mini"  # or gemini/gemini-2.0-flash

# Required env var for chosen provider
OPENAI_API_KEY=your-key  # for openai/*
GEMINI_API_KEY=your-key  # for gemini/*
```

Supported: `gemini/*`, `openai/*`, `anthropic/*`, `groq/*`, `mistral/*`

---

### Deployment Issues

**Q: Hugging Face build fails with "Permission denied"**

1. Check the secret name is exactly `GITHUB_PAT`
2. Verify the PAT has read access to the repo
3. Regenerate the PAT and update the HF secret

---

**Q: Docker build fails installing private package**

Pass the PAT as build argument:
```bash
docker build --build-arg GITHUB_PAT=ghp_xxxx -t my-agent .
```

In Dockerfile:
```dockerfile
ARG GITHUB_PAT
RUN pip install git+https://${GITHUB_PAT}@github.com/santhoshmallojwala/unifyriver.git@shivam
```

---

## Examples

### Simple Summarizer Agent

```
summarizer_agent/
├── agent.py
├── tools/
│   └── summarize.py
└── requirements.txt
```

### Invoice Processing Agent

```
invoice_agent/
├── agent.py
├── state.py
├── settings.py
├── api/main.py
├── prompts/templates.py
├── tools/
│   ├── extract_invoice.py
│   └── validate_invoice.py
└── matching/
    └── po_matcher.py
```

### LC Validation Agent (Complex)

```
lc_agent/
├── agent.py
├── state.py
├── settings.py
├── api/
├── prompts/
├── tools/
├── extraction/
├── validation/
├── integrations/
├── rag/
└── utils/
```

---

## API Reference (Quick Reference)

### Imports

```python
# Core
from point9_platform import BaseAgent, tool, ToolRegistry, UserSettings, SYSTEM_SETTINGS

# Storage (optional)
from point9_platform.storage import S3Storage, MongoStore, get_s3_storage, get_mongo_store

# Logging
from point9_platform.observability import setup_logging, get_logger

# Streaming
from point9_platform.observability.emitter import get_session_emitter, StepType, StepStatus

# LLM
from point9_platform.llm.provider import get_llm_provider

# Health endpoints
from point9_platform.health import create_health_router

# State utilities
from point9_platform.agent.state import message_reducer, DocumentInfo
```

### BaseAgent Methods

| Method | Required | Returns | Description |
|--------|----------|---------|-------------|
| `get_agent_name()` | ✅ | `str` | Unique agent identifier |
| `get_domain_keywords()` | ✅ | `List[str]` | Keywords for request validation |
| `create_initial_state(session_id)` | ✅ | `StateT` | Create initial TypedDict state |
| `get_prompts()` | ✅ | `Dict[str, str]` | Prompt templates |
| `get_state_class()` | No | `Type` | TypedDict class for state |
| `create_graph()` | No | `CompiledGraph` | Custom LangGraph workflow |
| `on_start(state)` | No | `None` | Lifecycle hook before processing |
| `on_complete(state, result)` | No | `None` | Lifecycle hook after success |
| `on_error(state, error)` | No | `None` | Lifecycle hook on failure |
| `process(message, documents)` | Built-in | `Dict` | Main entry point |

### @tool Decorator

```python
@tool(
    name="tool_name",           # Required: unique tool name
    description="What it does"  # Required: LLM uses this to decide when to call
)
def my_tool(
    param1: str,                # Regular parameters (sent to LLM schema)
    param2: int = 10,           # Optional with default
    state: Dict[str, Any] = None  # Injected automatically, not in schema
) -> Dict[str, Any]:
    return {"status": "success", "data": ...}
```

---

## License

Proprietary - Point9 Team

