"""
Custom wrapper for Anthropic client.

This module provides a wrapper for Anthropic client initialization
and ensures a clean setup for Claude API access.
"""

import logging
import os
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def _apply_anthropic_patches():
    """Apply monkey patches to fix Anthropic client initialization issues."""
    try:
        logger.info("Applying Anthropic monkey patches")
        
        # Import anthropic modules
        from anthropic._base_client import SyncHttpxClientWrapper
        
        # Save the original __init__ method
        original_init = SyncHttpxClientWrapper.__init__
        
        # Define a new init method that filters out 'proxies'
        def patched_init(self, *args, **kwargs):
            # Filter out problematic parameters
            if 'proxies' in kwargs:
                logger.warning("Removing 'proxies' parameter from httpx client initialization")
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

def create_anthropic_client(api_key: Optional[str] = None):
    """
    Create a clean Anthropic client for Claude API access.
    
    Args:
        api_key: Optional API key (uses ANTHROPIC_API_KEY env var if not provided)
    
    Returns:
        Anthropic client instance
    """
    try:
        # Apply the monkey patching to fix the 'proxies' issue
        _apply_anthropic_patches()
        
        # Import anthropic after applying patches
        import anthropic
        
        # Get the API key if not provided
        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("No API key provided and ANTHROPIC_API_KEY environment variable not set")
        
        # Create the client (our patches should handle any proxies issues)
        client = anthropic.Anthropic(api_key=api_key)
            
        logger.info("Successfully created Anthropic client")
        return client
    except Exception as e:
        logger.error(f"Error creating Anthropic client: {str(e)}")
        raise