"""
Monkey patch for OpenAI and Anthropic API to fix proxies issue.

This module applies monkey patches to fix client initialization
issues with the 'proxies' parameter for both OpenAI and Anthropic.
It also provides utility functions for checking mock mode configuration.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def is_mock_mode() -> bool:
    """Check if mock mode is enabled via environment variable."""
    mock_mode_env = os.environ.get("CDAS_MOCK_MODE", "0").lower()
    result = mock_mode_env in ("1", "true", "yes", "on")
    logger.info(f"Checking mock mode: CDAS_MOCK_MODE={mock_mode_env}, result={result}")
    return result

def check_mock_mode_config(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Determine if mock mode should be used based on environment and config.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        bool: True if mock mode should be used, False otherwise
    """
    # Configuration order of precedence:
    # 1. Explicit config parameter (if provided)
    # 2. Environment variable
    
    if config is not None and 'mock_mode' in config:
        return config['mock_mode']
    
    return is_mock_mode()

def apply_openai_patches():
    """Apply monkey patches to OpenAI package."""
    try:
        logger.info("Applying OpenAI monkey patches")
        
        # Import the OpenAI base client
        from openai._base_client import BaseClient, SyncHttpxClientWrapper
        
        # Save the original __init__ method
        original_init = SyncHttpxClientWrapper.__init__
        
        # Define our patched init that filters out 'proxies'
        def patched_init(self, **kwargs):
            # Remove problematic 'proxies' parameter if present
            if 'proxies' in kwargs:
                logger.warning("Removing 'proxies' parameter from httpx client initialization")
                del kwargs['proxies']
            
            # Call the original init with the cleaned kwargs
            return original_init(self, **kwargs)
        
        # Apply our patch
        SyncHttpxClientWrapper.__init__ = patched_init
        
        logger.info("Successfully applied OpenAI monkey patches")
        return True
    except Exception as e:
        logger.error(f"Failed to apply OpenAI patches: {str(e)}")
        return False

def apply_anthropic_patches():
    """Apply monkey patches to Anthropic package."""
    try:
        logger.info("Applying Anthropic monkey patches")
        
        # Import the Anthropic client
        from anthropic import Client
        
        # Save the original __init__ method
        original_init = Client.__init__
        
        # Define our patched init that filters out 'proxies'
        def patched_init(self, *args, **kwargs):
            # Remove problematic 'proxies' parameter if present
            if 'proxies' in kwargs:
                logger.warning("Removing 'proxies' parameter from Anthropic client initialization")
                del kwargs['proxies']
            
            # Call the original init with the cleaned kwargs
            return original_init(self, *args, **kwargs)
        
        # Apply our patch
        Client.__init__ = patched_init
        
        logger.info("Successfully applied Anthropic monkey patches")
        return True
    except Exception as e:
        logger.error(f"Failed to apply Anthropic patches: {str(e)}")
        return False