#!/usr/bin/env python3
"""
Test script for LLMManager with mock mode.
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to Python path to ensure imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import CDAS modules
from cdas.ai.llm import LLMManager

def main():
    """Test the LLMManager with mock mode."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Get OpenAI API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    logger.info(f"Found API key starting with: {api_key[:10] if api_key else 'None'}...")
    
    # Create LLMManager with mock mode
    logger.info("Creating LLMManager with mock mode enabled...")
    config = {
        "provider": "openai",
        "model": "o4-mini",
        "api_key": api_key,
        "mock_mode": True,  # Enable mock mode
        "reasoning_effort": "medium"
    }
    
    llm_manager = LLMManager(config)
    
    # Test simple text generation
    logger.info("\nTesting simple text generation...")
    prompt = "Explain in 3-4 sentences how construction change orders can lead to disputes."
    response = llm_manager.generate(
        prompt=prompt,
        system_prompt="You are an expert in construction law and finance."
    )
    logger.info(f"Response: {response}")
    
    # Test text generation with different prompt types
    logger.info("\nTesting with different prompt types...")
    
    # Summarization prompt
    summary_prompt = "Summarize the key reasons why construction projects exceed their budgets."
    summary_response = llm_manager.generate(prompt=summary_prompt)
    logger.info(f"Summary response: {summary_response}")
    
    # Explanation prompt
    explain_prompt = "Explain the difference between a change order and a construction claim."
    explain_response = llm_manager.generate(prompt=explain_prompt)
    logger.info(f"Explanation response: {explain_response}")
    
    # Test with function calling
    logger.info("\nTesting function calling...")
    tools = [{
        "type": "function",
        "function": {
            "name": "extract_amounts",
            "description": "Extract monetary amounts from text",
            "parameters": {
                "type": "object",
                "properties": {
                    "amounts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "amount": {"type": "number"},
                                "currency": {"type": "string"},
                                "context": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["amounts"]
            }
        }
    }]
    
    tools_prompt = "Extract all monetary amounts from the following text: 'The contractor billed $45,000 for electrical work and $12,500 for plumbing repairs. The change order added an additional $8,750 for unforeseen conditions.'"
    tools_response = llm_manager.generate_with_tools(prompt=tools_prompt, tools=tools)
    
    logger.info(f"Function calling content: {tools_response['content']}")
    logger.info(f"Function calling tool_calls: {tools_response['tool_calls']}")
    
    logger.info("\n✅ LLMManager mock mode test completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)