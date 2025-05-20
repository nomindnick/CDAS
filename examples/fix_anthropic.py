#!/usr/bin/env python
"""
Fix for Anthropic client issue.

This script applies monkey patching to fix the proxies parameter issue
with the Anthropic client.
"""

import logging
import sys
import types
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    logger.error("No Anthropic API key found. Please set the ANTHROPIC_API_KEY environment variable.")
    api_key = "your_anthropic_api_key_here"  # This won't work but will let us test initialization

def apply_anthropic_patches():
    """Apply monkey patches to fix Anthropic client."""
    try:
        logger.info("Applying Anthropic monkey patches")
        
        # Import anthropic modules
        import anthropic
        from anthropic._base_client import SyncHttpxClientWrapper
        
        # Save the original __init__ method
        original_init = SyncHttpxClientWrapper.__init__
        
        # Define a new init method that filters out 'proxies'
        def patched_init(self, *args, **kwargs):
            # Filter out problematic parameters
            if 'proxies' in kwargs:
                logger.warning(f"Removing 'proxies' parameter from httpx client initialization")
                del kwargs['proxies']
            
            # Call original init with cleaned kwargs
            return original_init(self, *args, **kwargs)
        
        # Apply the patch
        SyncHttpxClientWrapper.__init__ = patched_init
        
        logger.info("Successfully applied Anthropic patches")
        return True
    except Exception as e:
        logger.error(f"Failed to apply Anthropic patches: {str(e)}")
        return False

def test_anthropic_client():
    """Test creating the Anthropic client."""
    try:
        # Apply the patches
        apply_anthropic_patches()
        
        # Import anthropic
        import anthropic
        
        # Create client
        logger.info("Creating Anthropic client")
        client = anthropic.Anthropic(api_key=api_key)
        logger.info("Successfully created Anthropic client")
        
        # Attempt to make a call (will likely fail with a placeholder key)
        try:
            logger.info("Attempting to call API (may fail with invalid key)")
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                messages=[
                    {"role": "user", "content": "Hello Claude! Please respond with a short greeting."}
                ],
                max_tokens=100
            )
            logger.info(f"API Response: {response.content[0].text}")
        except Exception as e:
            logger.error(f"API call failed (expected with placeholder key): {str(e)}")
        
        return client
    except Exception as e:
        logger.error(f"Error creating Anthropic client: {str(e)}")
        return None

if __name__ == "__main__":
    test_anthropic_client()