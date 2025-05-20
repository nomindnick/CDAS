#!/usr/bin/env python
"""
Simple script to test Anthropic API connectivity.

This script tests direct connection to the Anthropic API
without going through our custom wrapper. This helps isolate
if the issue is with our wrapper or with the Anthropic package itself.
"""

import os
import sys
from dotenv import load_dotenv

# Configure basic logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple_anthropic_test")

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    logger.error("No Anthropic API key found. Please set the ANTHROPIC_API_KEY environment variable.")
    sys.exit(1)

def test_direct_anthropic():
    """Test direct connection to Anthropic API."""
    try:
        # Add project root to path
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        
        # Import our custom wrapper
        from cdas.ai.anthropic_wrapper import create_anthropic_client
        
        logger.info("Creating Anthropic client with API key only")
        client = create_anthropic_client(api_key=api_key)
        
        # Test with a simple message
        logger.info("Sending a test message to Claude")
        
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            messages=[
                {"role": "user", "content": "Hello Claude! Please respond with a short greeting."}
            ],
            max_tokens=100
        )
        
        print("\n=== Direct Anthropic API Response ===")
        print(response.content[0].text)
        print("===================================\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Error with direct Anthropic test: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    test_direct_anthropic()