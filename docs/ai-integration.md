# AI Integration

The Construction Document Analysis System (CDAS) includes robust AI integration capabilities for enhancing document analysis, pattern detection, and natural language querying of construction documents.

## Overview

The AI integration module (`cdas.ai`) provides the following components:

- **LLMManager**: A unified interface for interacting with language models from different providers
- **EmbeddingManager**: Manages document embeddings for semantic search capabilities
- **InvestigatorAgent**: Agent-based approach to investigate financial discrepancies
- **Prompt Templates**: Specialized prompts for different analysis tasks
- **Tools**: Function-based tools that agents can use to gather information

## Configuration

AI features are configured via environment variables or a config file. The system supports both OpenAI and Anthropic models:

```bash
# .env file example for OpenAI
OPENAI_API_KEY=your_openai_api_key_here
AI_PROVIDER=openai
AI_LLM_MODEL=o4-mini
AI_EMBEDDING_MODEL=text-embedding-3-small
AI_REASONING_EFFORT=medium

# .env file example for Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here
AI_PROVIDER=anthropic
AI_LLM_MODEL=claude-3-7-sonnet-20250219
AI_TEMPERATURE=0.0
```

Configuration options:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `AI_PROVIDER` | LLM provider: "openai" or "anthropic" | openai |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI) | None |
| `ANTHROPIC_API_KEY` | Anthropic API key (if using Anthropic) | None |
| `AI_LLM_MODEL` | Language model to use | o4-mini (OpenAI) |
| `AI_EMBEDDING_MODEL` | Embedding model for semantic search | text-embedding-3-small |
| `AI_REASONING_EFFORT` | For o4 models: low, medium, high | medium |
| `AI_TEMPERATURE` | Temperature for Claude models | 0.0 |

## Components

### LLMManager

The LLM Manager provides a unified interface for working with language models:

```python
from cdas.ai.llm import LLMManager
from cdas.config import get_config

config = get_config()
llm_manager = LLMManager(config["ai"])

# Generate text
response = llm_manager.generate(
    prompt="Explain how change orders can lead to disputes.",
    system_prompt="You are an expert in construction law."
)

# Generate with function calling
response = llm_manager.generate_with_tools(
    prompt="Analyze this change order for potential issues.",
    tools=[...],  # List of tool definitions
    system_prompt="You are a financial investigator."
)
```

### EmbeddingManager

The Embedding Manager handles document embeddings for semantic search:

```python
from cdas.ai.embeddings import EmbeddingManager
from cdas.db.session import get_session

session = get_session()
embedding_manager = EmbeddingManager(session, config["ai"])

# Generate embeddings for a text
embeddings = embedding_manager.generate_embeddings(
    "Construction change order requesting additional payment."
)

# Embed a document (stores embeddings in database)
embedding_manager.embed_document("doc_123")

# Search for similar documents
results = embedding_manager.search(
    "foundation issues extra work", 
    limit=5
)
```

### InvestigatorAgent

The Investigator Agent helps analyze financial data and documents:

```python
from cdas.ai.agents.investigator import InvestigatorAgent

investigator = InvestigatorAgent(session, llm_manager)

# Investigate a question
investigation = investigator.investigate(
    "What evidence suggests the contractor double-billed for HVAC equipment?"
)

# Access results
steps = investigation['investigation_steps']
report = investigation['final_report']
```

## CLI Integration

The AI capabilities are integrated with the CLI:

```bash
# Ask a question about the project
python -m cdas.cli query ask "When was the first time the contractor billed for elevator maintenance?"

# Search for documents semantically
python -m cdas.cli query search "foundation issues" --semantic

# Generate a report with AI assistance
python -m cdas.cli report generate "billing_analysis" --output report.md
```

## Development

For developing extensions to the AI integration:

1. Custom Tools:
   - Create new tools in `cdas/ai/tools/`
   - Register tools in `cdas/ai/agents/tool_registry.py`

2. Custom Agents:
   - Create new agent classes in `cdas/ai/agents/`
   - Implement the agent loop pattern

3. Custom Prompts:
   - Add new prompt templates in `cdas/ai/prompts/`

## Mock Mode

The AI components support a robust "mock mode" for development and testing without requiring a valid API key or making actual API calls:

```python
# Enable mock mode in configuration
llm_config = {
    'provider': 'openai',
    'model': 'o4-mini',
    'api_key': api_key,
    'mock_mode': True  # Enable mock mode explicitly
}

llm_manager = LLMManager(llm_config)

# Components will also automatically fall back to mock mode if API calls fail
```

### Mock Mode Features

When in mock mode:

1. **LLMManager** will return predefined responses based on the prompt content:
   - Different responses for different prompt types (explanations, summaries, etc.)
   - Consistency between identical prompts
   - Support for both regular generation and tool-based generation

2. **EmbeddingManager** will generate consistent pseudorandom vectors:
   - Fixed-dimension vectors (1536 dimensions)
   - Deterministic output based on input text (same text = same embedding)
   - Normalized vectors suitable for similarity comparisons

3. **InvestigatorAgent** will simulate a complete investigation:
   - Mock investigation steps
   - Mock tool calls and results
   - Complete mock investigation report

### When to Use Mock Mode

Mock mode is useful for:
- Development without valid API credentials
- Unit and integration testing
- Demonstrations and presentations
- Reducing API costs during development
- CI/CD pipelines

### Automatic Fallback

Components will automatically switch to mock mode if:
- API initialization fails
- API calls return authentication or authorization errors
- Rate limits are exceeded

This behavior ensures graceful degradation even in production environments.

### Troubleshooting

#### Proxies Issue

If you encounter a "proxies" error when initializing the OpenAI client:

```
TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
```

The system includes a monkey patch to fix this issue. Apply it before creating any OpenAI clients:

```python
from cdas.ai.monkey_patch import apply_openai_patches
apply_openai_patches()
```

This patch handles a compatibility issue with OpenAI API client initialization in certain environments.

Test files are available in `tests/test_ai/`.

## Requirements

- OpenAI API access for language model and embeddings
- Database storage for document embeddings
- Python packages:
  - openai>=1.79.0  # Updated version to support o4-mini with reasoning_effort
  - tiktoken>=0.5.2
  - numpy>=1.26.3