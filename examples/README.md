# CDAS Example Scripts

This directory contains example scripts for demonstrating and testing various components of the Construction Document Analysis System (CDAS).

## AI Integration Examples

### Anthropic Claude Integration

- `test_anthropic_connection.py`: Tests the basic connection to Anthropic's Claude API.
- `test_anthropic_tools.py`: Demonstrates tool-calling capabilities with Claude.
- `ai_integration_example.py`: Comprehensive demonstration of AI integration using Claude for text generation.

### OpenAI Integration (Used for Embeddings)

- `test_openai_connection.py`: Tests the basic connection to OpenAI API.
- `test_openai_models.py`: Lists available models from the OpenAI API.
- `verify_api_key.py`: Verifies that the OpenAI API key is correctly configured.

### Component Testing

- `test_llm_manager.py`: Demonstrates the LLMManager component with mock mode support.
- `test_embedding_manager.py`: Shows the EmbeddingManager component with mock mode support.

## Running the Examples

1. Ensure you have set up the environment (see `CLAUDE.md` for instructions).
2. Make sure you have a `.env` file with the appropriate configuration:
   ```
   # Primary LLM (Anthropic Claude)
   ANTHROPIC_API_KEY=your-anthropic-api-key
   
   # For embeddings (still using OpenAI)
   OPENAI_API_KEY=your-openai-api-key
   
   # Optional configuration
   AI_LLM_MODEL=claude-3-7-sonnet-20250219
   AI_EMBEDDING_MODEL=text-embedding-3-small
   ```
3. Run an example:
   ```bash
   # Activate the virtual environment
   source venv/bin/activate
   
   # Run the AI integration example
   python examples/ai_integration_example.py
   
   # Test Anthropic Claude connection
   python examples/test_anthropic_connection.py
   ```

## Provider Selection

The AI integration components support multiple LLM providers:

- **Anthropic Claude** (default): Used for text generation and tool usage
- **OpenAI**: Used for embeddings and can be used as an alternative LLM provider

You can specify the provider when initializing the LLMManager:

```python
# For Anthropic Claude
llm_manager = LLMManager({
    'provider': 'anthropic',
    'model': 'claude-3-7-sonnet-20250219',
    'api_key': os.getenv('ANTHROPIC_API_KEY')
})

# For OpenAI (alternative)
llm_manager = LLMManager({
    'provider': 'openai',
    'model': 'o4-mini',
    'api_key': os.getenv('OPENAI_API_KEY'),
    'reasoning_effort': 'medium'  # Only needed for OpenAI o4 models
})
```

## Mock Mode

All AI integration examples support mock mode, which provides simulated responses without requiring valid API keys. This is useful for development, testing, and demonstrations.

To enable mock mode explicitly, set `mock_mode: True` in the component configuration:

```python
config = {
    'provider': 'anthropic',  # or 'openai'
    'model': 'claude-3-7-sonnet-20250219',
    'mock_mode': True  # Enable mock mode
}
```

The components will also automatically fall back to mock mode if API calls fail due to invalid credentials or other errors.

## Troubleshooting

If you encounter issues with the APIs, check:

1. Your API keys are correctly set in the `.env` file
2. Your Anthropic or OpenAI account has access to the specified models
3. You're using the latest versions of the packages:
   - `anthropic>=0.19.1`
   - `openai>=1.79.0`

For more detailed information, see the full AI integration documentation in `ai-integration-spec.md`.