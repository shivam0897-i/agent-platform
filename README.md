# Point9 Agent Platform

**Version 1.0.0**

A foundation platform for building production-ready AI agents with LangGraph.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Platform Architecture](#platform-architecture)
4. [Creating an Agent](#creating-an-agent)
5. [Agent Structure](#agent-structure)
6. [Core Components](#core-components)
7. [Creating Tools](#creating-tools)
8. [Configuration](#configuration)
9. [API Endpoints](#api-endpoints)
10. [Multi-turn Chat](#multi-turn-chat)
11. [Progress Streaming](#progress-streaming)
12. [Deployment](#deployment)
13. [Examples](#examples)

---

## Installation

### For Development (Editable Install)

```bash
cd point9-agent-platform
pip install -e .
```

### For Production

```bash
# From Git
pip install git+https://github.com/your-org/point9-agent-platform.git

# Or from PyPI (if published)
pip install point9-agent-platform
```

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

### Hugging Face Spaces

Each agent deploys to its own Space:

```
# app.py (at root)
from api.main import app

# requirements.txt
point9-agent-platform @ git+https://github.com/your-org/point9-agent-platform.git
fastapi>=0.104.0
uvicorn>=0.24.0
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
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

## License

Proprietary - Point9 Team
