#!/usr/bin/env python
"""
Minimal script to test Anthropic API.

This script has minimal dependencies and performs a basic API call
to ensure everything is working correctly.
"""

import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    logger.error("No Anthropic API key found in environment variables")
    api_key = "your_anthropic_api_key_here"  # Replace with actual API key for testing
    logger.warning(f"Using placeholder API key: {api_key} - this won't work for real calls")

try:
    # Import anthropic
    import anthropic
    
    # Print version 
    logger.info(f"Using Anthropic SDK version: {anthropic.__version__}")
    
    # Create client directly
    client = anthropic.Anthropic(api_key=api_key)
    logger.info("Successfully created Anthropic client")
    
    # Verify we can reach the API (this will fail with invalid API key)
    # But at least we know client initialization works
    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            messages=[
                {"role": "user", "content": "Hello Claude! Please respond with a short greeting."}
            ],
            max_tokens=100
        )
        print(f"API Response: {response.content[0].text}")
    except Exception as e:
        logger.error(f"Error calling API (expected with placeholder key): {str(e)}")
        
except Exception as e:
    logger.error(f"Error with Anthropic client: {str(e)}")
    
print("Script completed.")