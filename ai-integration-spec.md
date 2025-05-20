# AI Integration Specification

## Overview

The AI Integration component enhances the Construction Document Analysis System with advanced natural language processing, semantic understanding, and agentic capabilities. By integrating large language models (LLMs) and other AI technologies, the system can intelligently analyze document content, understand context, and generate insights that would be difficult to achieve with rule-based approaches alone.

## Key Capabilities

1. **Document Understanding**: Extract meaning and context from unstructured text
2. **Semantic Search**: Find relevant information based on meaning rather than keywords
3. **Pattern Recognition**: Identify complex patterns in financial data and document relationships
4. **Anomaly Detection**: Detect unusual or suspicious patterns that may indicate fraud or errors
5. **Report Generation**: Create natural language summaries and explanations of findings
6. **Agentic Analysis**: Perform multi-step reasoning to investigate complex scenarios
7. **Interactive Querying**: Allow natural language querying of the document database

## Component Architecture

```
ai/
├─ __init__.py
├─ llm.py                   # LLM integration
├─ embeddings.py            # Document embeddings
├─ anthropic_wrapper.py     # Anthropic Claude API wrapper
├─ openai_wrapper.py        # OpenAI API wrapper
├─ monkey_patch.py          # Monkey patches for API compatibility
├─ prompts/                 # Prompt templates
│   ├─ __init__.py
│   ├─ document_analysis.py # Document analysis prompts
│   ├─ financial_analysis.py # Financial analysis prompts
│   └─ report_generation.py # Report generation prompts
├─ agents/                  # Agent implementations
│   ├─ __init__.py
│   ├─ investigator.py      # Financial investigation agent
│   ├─ reporter.py          # Report generation agent
│   └─ tool_registry.py     # Agent tool registry
├─ tools/                   # Tool implementations
│   ├─ __init__.py
│   ├─ document_tools.py    # Document retrieval tools
│   ├─ financial_tools.py   # Financial analysis tools
│   └─ database_tools.py    # Database query tools
└─ semantic_search/         # Semantic search capabilities
    ├─ __init__.py
    ├─ vectorizer.py        # Text vectorization
    ├─ index.py             # Vector index management
    └─ search.py            # Semantic search implementation
```

## Core Classes

### LLMManager

Manages interactions with language models (supports both Anthropic Claude and OpenAI).

```python
class LLMManager:
    """Manages interactions with language models."""
    
    def __init__(self, config=None):
        """Initialize the LLM manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.mock_mode = self.config.get('mock_mode', False)
        
        # Initialize client based on provider
        self.provider = self.config.get('provider', 'anthropic')
        self.client = None
        
        # Set provider-specific defaults
        if self.provider == 'anthropic':
            self.model = self.config.get('model', 'claude-3-7-sonnet-20250219') 
            self.api_key = self.config.get('api_key')
            self.temperature = self.config.get('temperature', 0.0)
        elif self.provider == 'openai':
            self.model = self.config.get('model', 'o4-mini')
            self.api_key = self.config.get('api_key')
            self.temperature = self.config.get('temperature', 0.0)
            self.reasoning_effort = self.config.get('reasoning_effort', 'medium')
        
        if self.mock_mode:
            logger.warning(
                "Initializing LLMManager in MOCK MODE. "
                "API calls will be simulated and no actual requests will be made."
            )
        elif self.provider == 'anthropic':
            try:
                # Use our Anthropic wrapper
                from cdas.ai.anthropic_wrapper import create_anthropic_client
                self.client = create_anthropic_client(api_key=self.api_key)
                logger.info("Initialized Anthropic client")
            except ImportError:
                logger.error("Anthropic package not installed. Please install with: pip install anthropic")
                raise
            except Exception as e:
                logger.error(f"Error initializing Anthropic client: {str(e)}")
                logger.warning("Falling back to MOCK MODE due to initialization error")
                self.mock_mode = True
        elif self.provider == 'openai':
            try:
                # Use our custom wrapper to avoid the proxies issue
                from cdas.ai.openai_wrapper import create_openai_client
                self.client = create_openai_client(api_key=self.api_key)
                logger.info("Initialized OpenAI client")
            except ImportError:
                logger.error("OpenAI package not installed. Please install with: pip install openai")
                raise
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {str(e)}")
                logger.warning("Falling back to MOCK MODE due to initialization error")
                self.mock_mode = True
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def generate(self, prompt, system_prompt=None, temperature=None, reasoning_effort=None):
        """Generate text from the language model.
        
        Args:
            prompt: Prompt text
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            reasoning_effort: Optional reasoning effort for OpenAI o4 models ('low', 'medium', 'high')
            
        Returns:
            Generated text
        """
        # Check if we're in mock mode
        if self.mock_mode:
            logger.info("Running in MOCK MODE - generating simulated response")
            
            # Basic mock response based on the prompt
            if system_prompt and "expert" in system_prompt.lower():
                return f"[MOCK EXPERT RESPONSE] Response to: {prompt[:50]}..."
            elif "summarize" in prompt.lower() or "summary" in prompt.lower():
                return f"[MOCK SUMMARY] Here is a concise summary of the requested information."
            elif "explain" in prompt.lower():
                return f"[MOCK EXPLANATION] Here is a detailed explanation about {prompt[:50]}..."
            else:
                return f"[MOCK RESPONSE] This is a simulated response to your prompt: {prompt[:50]}..."
                
        # Real API usage
        if not self.client:
            raise ValueError("LLM client not initialized")
            
        temp = temperature if temperature is not None else self.temperature
        
        try:
            if self.provider == 'anthropic':
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=4000
                )
                
                return response.content[0].text
            
            elif self.provider == 'openai':
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                # Parameters for the API call
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temp
                }
                
                # Add reasoning_effort parameter for o4 models
                if reasoning_effort is not None and self.model.startswith('o4-'):
                    params["reasoning_effort"] = reasoning_effort
                
                response = self.client.chat.completions.create(**params)
                
                return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            # Fall back to mock mode if API call fails
            logger.warning("Falling back to MOCK MODE due to API error")
            self.mock_mode = True
            # Recursive call - now will use the mock mode
            return self.generate(prompt, system_prompt, temperature, reasoning_effort)
    
    def generate_with_tools(self, prompt, tools, system_prompt=None, temperature=None, reasoning_effort=None):
        """Generate text with function calling capabilities.
        
        Args:
            prompt: Prompt text
            tools: List of tool definitions
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            reasoning_effort: Optional reasoning effort for o4 models ('low', 'medium', 'high')
            
        Returns:
            Generated text and any tool calls
        """
        # Check if we're in mock mode
        if self.mock_mode:
            logger.info("Running in MOCK MODE - generating simulated tool response")
            
            # Create a mock tool call if there are tools defined
            mock_tool_calls = None
            if tools and len(tools) > 0:
                # Get the first defined function for our mock
                first_tool = tools[0]
                function_name = first_tool.get('function', {}).get('name', 'unknown_function')
                
                # Create a simple mock tool call
                mock_tool_calls = [{
                    "id": "call_mock123456",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": '{"mock_param": "mock_value"}'
                    }
                }]
            
            # Basic mock content based on the prompt
            if "analyze" in prompt.lower():
                mock_content = "[MOCK ANALYSIS] Here's my analysis of the requested information."
            elif "extract" in prompt.lower():
                mock_content = "[MOCK EXTRACTION] I've extracted the key information requested."
            else:
                mock_content = f"[MOCK TOOL RESPONSE] This is a simulated response to: {prompt[:50]}..."
            
            return {
                'content': mock_content,
                'tool_calls': mock_tool_calls
            }
                
        # Real API usage
        if not self.client:
            raise ValueError("LLM client not initialized")
            
        temp = temperature if temperature is not None else self.temperature
        
        try:
            if self.provider == 'anthropic':
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                # Use beta.tools endpoint for tool usage with Anthropic
                try:
                    # Convert OpenAI-style tools format to Anthropic format
                    anthropic_tools = []
                    for tool in tools:
                        if 'function' in tool:
                            function_info = tool['function']
                            anthropic_tools.append({
                                "name": function_info['name'],
                                "description": function_info.get('description', ''),
                                "input_schema": {
                                    "type": "object",
                                    "properties": function_info.get('parameters', {}).get('properties', {}),
                                    "required": function_info.get('parameters', {}).get('required', [])
                                }
                            })
                    
                    response = self.client.beta.tools.messages.create(
                        model=self.model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=4000,
                        tools=anthropic_tools
                    )
                    
                    # Extract tool calls if present
                    tool_calls = None
                    content = None
                    
                    if response.content:
                        for item in response.content:
                            if item.type == 'tool_use':
                                if tool_calls is None:
                                    tool_calls = []
                                
                                # Convert Anthropic tool call format to OpenAI-like format for compatibility
                                tool_calls.append({
                                    "id": item.id,
                                    "type": "function",
                                    "function": {
                                        "name": item.name,
                                        "arguments": json.dumps(item.input) if isinstance(item.input, dict) else item.input
                                    }
                                })
                            elif item.type == 'text':
                                content = item.text
                    
                    return {
                        'content': content,
                        'tool_calls': tool_calls
                    }
                    
                except Exception as e:
                    logger.error(f"Error with Anthropic tools API: {str(e)}")
                    logger.warning("Tool calling failed with Anthropic, falling back to standard message API")
                    
                    # Fallback to standard message API without tools
                    response = self.client.messages.create(
                        model=self.model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=4000
                    )
                    
                    return {
                        'content': response.content[0].text if response.content else None,
                        'tool_calls': None
                    }
                
            elif self.provider == 'openai':
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                # Parameters for the API call
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temp,
                    "tools": tools
                }
                
                # Add reasoning_effort parameter for o4 models
                if reasoning_effort is not None and self.model.startswith('o4-'):
                    params["reasoning_effort"] = reasoning_effort
                
                response = self.client.chat.completions.create(**params)
                
                return {
                    'content': response.choices[0].message.content,
                    'tool_calls': response.choices[0].message.tool_calls if hasattr(response.choices[0].message, 'tool_calls') else None
                }
            else:
                # For providers without native function calling
                raise ValueError(f"Tool calling not implemented for provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Error generating text with tools: {str(e)}")
            # Fall back to mock mode if API call fails
            logger.warning("Falling back to MOCK MODE due to API error")
            self.mock_mode = True
            # Recursive call - now will use the mock mode
            return self.generate_with_tools(prompt, tools, system_prompt, temperature, reasoning_effort)
```

### AnthropicWrapper

Provides a clean wrapper for creating Anthropic clients.

```python
def create_anthropic_client(api_key=None):
    """
    Create a clean Anthropic client for Claude API access.
    
    Args:
        api_key: Optional API key (uses ANTHROPIC_API_KEY env var if not provided)
    
    Returns:
        Anthropic client instance
    """
    try:
        from anthropic import Anthropic
        
        # Only pass the API key and nothing else
        if api_key:
            client = Anthropic(api_key=api_key)
        else:
            client = Anthropic()
            
        logger.info("Successfully created Anthropic client")
        return client
    except Exception as e:
        logger.error(f"Error creating Anthropic client: {str(e)}")
        raise
```

### EmbeddingManager

Manages document embeddings for semantic search (using OpenAI embeddings).

```python
class EmbeddingManager:
    """Manages document embeddings for semantic search."""
    
    def __init__(self, db_session, config=None):
        """Initialize the embedding manager.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Initialize embedding model
        self.embedding_model = self.config.get('embedding_model', 'text-embedding-3-small')
        self.api_key = self.config.get('api_key')
        self.mock_mode = self.config.get('mock_mode', False)
        self.client = None
        
        if self.mock_mode:
            logger.warning(
                "Initializing EmbeddingManager in MOCK MODE. "
                "API calls will be simulated and no actual requests will be made."
            )
        else:
            try:
                # Use our custom wrapper to avoid the proxies issue
                from cdas.ai.openai_wrapper import create_openai_client
                self.client = create_openai_client(api_key=self.api_key)
                logger.info("Initialized OpenAI client for embeddings")
            except ImportError:
                logger.error("OpenAI package not installed. Please install with: pip install openai")
                raise
            except Exception as e:
                logger.error(f"Error initializing OpenAI client for embeddings: {str(e)}")
                logger.warning("Falling back to MOCK MODE due to initialization error")
                self.mock_mode = True
    
    def generate_embeddings(self, text):
        """Generate embeddings for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Check if we're in mock mode
        if self.mock_mode:
            logger.info("Running in MOCK MODE - generating simulated embeddings")
            
            # Generate fixed-length mock embedding
            # Using a simple hash-based approach to ensure consistent vectors for the same text
            import numpy as np
            import hashlib
            
            # Create a hash of the text
            text_hash = hashlib.md5(text.encode()).hexdigest()
            # Use just the first 8 chars of hash to stay within numpy's seed range (0 to 2^32-1)
            hash_int = int(text_hash[:8], 16)
            
            # Use the hash to seed random number generator for reproducibility
            np.random.seed(hash_int)
            
            # Generate a mock embedding vector - use 1536 dimensions as that's standard for many embedding models
            dimensions = 1536
            mock_embedding = list(np.random.rand(dimensions) - 0.5)  # Center around 0
            
            # Normalize the vector to unit length (common for embeddings)
            norm = np.linalg.norm(mock_embedding)
            if norm > 0:
                mock_embedding = [x / norm for x in mock_embedding]
                
            logger.info(f"Generated mock embedding with {dimensions} dimensions")
            return mock_embedding
        
        # Real API usage        
        if not self.client:
            raise ValueError("Embedding client not initialized")
            
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Fall back to mock mode if API call fails
            logger.warning("Falling back to MOCK MODE due to API error")
            self.mock_mode = True
            # Recursive call - now will use the mock mode
            return self.generate_embeddings(text)
    
    # ... other methods (embed_document, search) ...
```

### InvestigatorAgent

Agent that investigates financial discrepancies.

```python
class InvestigatorAgent:
    """Agent that investigates financial discrepancies."""
    
    def __init__(self, db_session, llm_manager, config=None):
        """Initialize the investigator agent.
        
        Args:
            db_session: Database session
            llm_manager: LLM manager
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.llm = llm_manager
        self.config = config or {}
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
    
    def investigate(self, question, context=None):
        """Investigate a question about the construction dispute.
        
        Args:
            question: Question to investigate
            context: Optional context information
            
        Returns:
            Investigation results
        """
        # Check if we're in mock mode (via LLM manager)
        is_mock_mode = getattr(self.llm, 'mock_mode', False)
        
        # If in mock mode, return a simulated investigation right away
        if is_mock_mode:
            logger.info("In mock mode, generating simplified mock investigation")
            
            # Mock steps for the investigation
            mock_steps = [
                "Analyzing documents related to potential double-billing instances...",
                "Identifying suspicious patterns in billing amounts across documents...",
                "Cross-referencing payment applications with change orders...",
                "Verifying item descriptions and amounts in sequential documents..."
            ]
            
            # Generate mock final report
            final_report = self._generate_final_report(question, context, mock_steps)
            
            return {
                'investigation_steps': mock_steps,
                'final_report': final_report
            }
        
        # Regular investigation flow
        # Prepare initial prompt
        prompt = self._prepare_prompt(question, context)
        
        # Run agent loop
        max_iterations = self.config.get('max_iterations', 10)
        full_response = []
        
        try:
            for i in range(max_iterations):
                logger.info(f"Investigation iteration {i+1}/{max_iterations}")
                
                # Generate response with tools
                response = self.llm.generate_with_tools(prompt, self.tools, self.system_prompt)
                
                # Check for tool calls
                if response.get('tool_calls'):
                    # Execute tool calls
                    tool_results = self._execute_tool_calls(response['tool_calls'])
                    
                    # Add tool results to prompt
                    tool_results_text = "\n".join([f"Tool: {tool}\nResult: {result}" for tool, result in tool_results.items()])
                    prompt += f"\n\nTool Results:\n{tool_results_text}\n\nContinue your investigation:"
                else:
                    # No more tool calls, we're done
                    if response['content']:
                        full_response.append(response['content'])
                    break
                
                # Add response content to full response
                if response['content']:
                    full_response.append(response['content'])
            
            # Generate final report
            final_report = self._generate_final_report(question, context, full_response)
            
            return {
                'investigation_steps': full_response,
                'final_report': final_report
            }
        
        except Exception as e:
            logger.error(f"Error during investigation: {str(e)}")
            return {
                'investigation_steps': full_response,
                'error': str(e)
            }
    
    def _execute_tool_calls(self, tool_calls):
        """Execute tool calls and return results."""
        results = {}
        
        # Check if we're in mock mode (via LLM manager)
        is_mock_mode = getattr(self.llm, 'mock_mode', False)
        
        # Handle mock mode differently
        if is_mock_mode:
            logger.info("In mock mode, generating mock tool results")
            
            for tool_call in tool_calls:
                # In mock mode, tool_call might not have function attribute directly
                # It might be a dict instead of an object
                if isinstance(tool_call, dict):
                    function_name = tool_call.get('function', {}).get('name', 'unknown_function')
                    arguments_str = tool_call.get('function', {}).get('arguments', '{}')
                    try:
                        arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                    except:
                        arguments = {}
                else:
                    # Regular OpenAI response format
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                
                logger.info(f"Mock executing tool call: {function_name} with arguments: {arguments}")
                
                # Generate mock results based on function name
                if function_name == "search_documents":
                    results[function_name] = "[MOCK] Found 3 documents matching the criteria related to billing disputes."
                elif function_name == "search_line_items":
                    results[function_name] = "[MOCK] Found 5 line items with similar amounts across different documents."
                elif function_name == "find_suspicious_patterns":
                    results[function_name] = "[MOCK] Identified 2 suspicious patterns with recurring amounts in different documents."
                elif function_name == "run_sql_query":
                    results[function_name] = "[MOCK] Query returned 10 rows showing potential duplicate billing entries."
                else:
                    results[function_name] = f"[MOCK] Unknown tool: {function_name}"
            
            return results
        
        # Regular execution mode - handle different LLM providers
        provider = getattr(self.llm, 'provider', 'anthropic')
        
        for tool_call in tool_calls:
            # Extract function name and arguments based on provider format
            if provider == 'anthropic':
                # Handle Anthropic format
                if isinstance(tool_call, dict):
                    function_name = tool_call.get('function', {}).get('name')
                    arguments_str = tool_call.get('function', {}).get('arguments')
                    try:
                        arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                    except Exception as e:
                        logger.error(f"Error parsing tool call arguments: {str(e)}")
                        arguments = {}
                else:
                    # TypedObject format
                    function_name = getattr(getattr(tool_call, 'function', None), 'name', None)
                    arguments_str = getattr(getattr(tool_call, 'function', None), 'arguments', '{}')
                    try:
                        arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                    except Exception as e:
                        logger.error(f"Error parsing tool call arguments: {str(e)}")
                        arguments = {}
            else:
                # Handle OpenAI format
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
            
            logger.info(f"Executing tool call: {function_name} with arguments: {arguments}")
            
            # Execute the appropriate tool function
            if function_name == "search_documents":
                results[function_name] = self._search_documents(arguments)
            elif function_name == "search_line_items":
                results[function_name] = self._search_line_items(arguments)
            elif function_name == "find_suspicious_patterns":
                results[function_name] = self._find_suspicious_patterns(arguments)
            elif function_name == "run_sql_query":
                results[function_name] = self._run_sql_query(arguments)
            else:
                results[function_name] = f"Unknown tool: {function_name}"
        
        return results
    
    # ... other methods ...
```

## Mock Mode

The system includes a comprehensive mock mode that allows for development and testing without making actual API calls. This mode is particularly useful in the following situations:

1. **Development Environment**: When developing without API keys
2. **Automated Testing**: For integration and unit tests
3. **API Fallback**: When API calls fail or rate limits are reached
4. **Demo Environments**: For demonstrations without using API credits

Mock mode is implemented at several levels:

1. **LLMManager**: Generates mock responses based on prompt content
2. **EmbeddingManager**: Creates deterministic mock embeddings using a hash-based approach
3. **InvestigatorAgent**: Provides mock investigation steps and final reports

Mock responses are designed to be consistent for the same inputs, which helps with testing and debugging.

## API Configuration

### Anthropic Claude API

The system uses Anthropic's Claude API for text generation and tool usage:

```python
# Environment variables
ANTHROPIC_API_KEY=<your-api-key>

# Configuration
config = {
    'provider': 'anthropic',
    'model': 'claude-3-7-sonnet-20250219',  # You can use other Claude models
    'api_key': os.getenv('ANTHROPIC_API_KEY'),
    'temperature': 0.0  # Low temperature for deterministic outputs
}

# Initialize LLM Manager
llm_manager = LLMManager(config)
```

### OpenAI API for Embeddings

The system continues to use OpenAI's API for text embeddings:

```python
# Environment variables
OPENAI_API_KEY=<your-api-key>

# Configuration
config = {
    'embedding_model': 'text-embedding-3-small',
    'api_key': os.getenv('OPENAI_API_KEY')
}

# Initialize Embedding Manager
embedding_manager = EmbeddingManager(db_session, config)
```

## Model Selection

1. **Primary LLM**: Claude-3-7-Sonnet (Anthropic) for high-quality reasoning and tool usage
2. **Embedding Model**: text-embedding-3-small (OpenAI) for document embeddings
3. **Local Options**: Consider local models for privacy-sensitive deployments

## Implementation Guidelines

1. **API Security**: Secure API keys and credentials
2. **Resource Management**: Manage token usage and API costs
3. **Error Handling**: Implement robust error handling for API calls
4. **Caching**: Cache results to reduce API calls
5. **Fallbacks**: Implement fallbacks for API failures (using mock mode)
6. **Privacy**: Ensure sensitive document information is handled properly

## Testing Strategy

1. **Unit Tests**: Test individual AI components
2. **Integration Tests**: Test integration with other system components
3. **Prompt Tests**: Test effectiveness of prompts
4. **Scenario Tests**: Test complex scenarios
5. **Fallback Tests**: Test fallback mechanisms

## Ethical Considerations

1. **Bias Mitigation**: Ensure AI outputs are objective and unbiased
2. **Transparency**: Make AI contribution to analysis clear
3. **Human Oversight**: Maintain human review of AI-generated content
4. **Accountability**: Maintain traceability of AI-generated analysis