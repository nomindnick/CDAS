#!/usr/bin/env python
"""
Test script for Anthropic Claude tool usage.

This script demonstrates how to use Claude with tool calling capabilities.
It includes examples of defining tools and handling Claude's responses.
"""

import os
import logging
import sys
import json
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables from .env file
load_dotenv()

def test_anthropic_tools():
    """Test Anthropic Claude with tool calling capabilities."""
    from cdas.ai.llm import LLMManager
    
    # Get API key from environment variable
    api_key = os.getenv('ANTHROPIC_API_KEY')
    
    # Define mock functions for tools
    def get_weather(location):
        """Mock function to get weather for a location."""
        return f"The weather in {location} is currently 72°F and sunny."
    
    def search_database(query):
        """Mock function to search a database."""
        return f"Found 3 results for query: '{query}'"
    
    # Sample tool definitions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search the database for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    # Initialize LLM Manager with Anthropic
    try:
        llm = LLMManager({
            'provider': 'anthropic',
            'model': 'claude-3-7-sonnet-20250219',  # You can change this to a different Claude model
            'api_key': api_key,
            'mock_mode': False  # Set True to use mock mode instead of real API
        })
        
        # Test prompt that should trigger tool usage
        prompt = "What's the weather like in San Francisco? Also, can you search the database for information about recent construction projects?"
        
        # Generate response with tools
        response = llm.generate_with_tools(prompt, tools)
        
        print("\n=== Claude Response with Tools ===")
        print(f"Content: {response.get('content')}")
        print(f"Tool Calls: {json.dumps(response.get('tool_calls'), indent=2) if response.get('tool_calls') else 'None'}")
        print("=================================\n")
        
        # Process tool calls if present
        if response.get('tool_calls'):
            print("Processing tool calls...\n")
            
            # Execute each tool call
            tool_results = {}
            for tool_call in response.get('tool_calls', []):
                function_name = tool_call.get('function', {}).get('name')
                arguments_str = tool_call.get('function', {}).get('arguments')
                
                try:
                    arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                    
                    if function_name == 'get_weather':
                        location = arguments.get('location')
                        result = get_weather(location)
                    elif function_name == 'search_database':
                        query = arguments.get('query')
                        result = search_database(query)
                    else:
                        result = f"Unknown function: {function_name}"
                        
                    tool_results[function_name] = result
                    print(f"📢 {function_name}: {result}")
                except Exception as e:
                    print(f"Error executing {function_name}: {str(e)}")
                    tool_results[function_name] = f"Error: {str(e)}"
            
            print("\nTool execution complete.")
                
        print("\n✅ Successfully tested Claude with tools")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Claude with tools: {str(e)}")
        
        # Fall back to mock mode
        print("\nFalling back to mock mode...")
        
        llm = LLMManager({
            'provider': 'anthropic',
            'mock_mode': True
        })
        
        response = llm.generate_with_tools(prompt, tools)
        
        print("\n=== Mock Mode Response with Tools ===")
        print(f"Content: {response.get('content')}")
        print(f"Tool Calls: {json.dumps(response.get('tool_calls'), indent=2) if response.get('tool_calls') else 'None'}")
        print("=====================================\n")
        
        print("✅ Mock mode is working correctly as a fallback")
        return False

if __name__ == "__main__":
    test_anthropic_tools()