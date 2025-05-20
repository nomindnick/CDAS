#!/usr/bin/env python
"""
Test script for Anthropic Claude API integration.

This script tests basic connectivity with the Anthropic Claude API.
It also demonstrates how to enable mock mode as a fallback.
"""

import os
import logging
import sys
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

def test_anthropic_connection():
    """Test basic connectivity with the Anthropic Claude API."""
    from cdas.ai.llm import LLMManager
    
    # Get API key from environment variable
    api_key = os.getenv('ANTHROPIC_API_KEY')
    
    # Test with real API first
    try:
        # Initialize LLM Manager with Anthropic
        llm = LLMManager({
            'provider': 'anthropic',
            'model': 'claude-3-7-sonnet-20250219',  # You can change this to a different Claude model 
            'api_key': api_key,
            'mock_mode': False  # Explicitly set to False to test real API
        })
        
        # Test simple prompt
        prompt = "Hello Claude! Please respond with a short greeting to confirm the connection is working correctly."
        response = llm.generate(prompt)
        
        print("\n=== Anthropic Claude API Response ===")
        print(response)
        print("===================================\n")
        
        print("✅ Successfully connected to Anthropic Claude API")
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Anthropic Claude API: {str(e)}")
        
        # Try with mock mode as fallback
        print("\nTrying with mock mode as fallback...")
        
        llm = LLMManager({
            'provider': 'anthropic',
            'mock_mode': True
        })
        
        prompt = "Hello Claude! Please respond with a short greeting."
        response = llm.generate(prompt)
        
        print("\n=== Mock Mode Response ===")
        print(response)
        print("========================\n")
        
        print("✅ Mock mode is working correctly as a fallback")
        return False

if __name__ == "__main__":
    test_anthropic_connection()