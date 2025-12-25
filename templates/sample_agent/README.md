# Sample Agent

Reference implementation of an agent using the platform.

Copy this folder as a starting point for your own agents.

## Structure

```
sample_agent/
├── agent.py        # Main agent class (extends BaseAgent)
├── state.py        # Domain-specific state
├── settings.py     # Domain-specific settings
├── tools/          # Agent-specific tools
└── prompts/        # Agent-specific prompts
```

## Usage

```python
from sample_agent.agent import DocumentAgent

agent = DocumentAgent(session_id="test-123")
result = agent.process("Extract data from the uploaded document")
```
