"""
Custom wrapper for OpenAI client to fix proxies issue.

This module provides a clean wrapper for OpenAI client initialization that
ensures no problematic parameters like 'proxies' are passed to the API.
"""

import logging
import os
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Apply monkey patches for OpenAI
from cdas.ai.monkey_patch import apply_openai_patches
apply_openai_patches()

def create_openai_client(api_key: Optional[str] = None):
    """
    Create a clean OpenAI client without any problematic parameters.
    
    Args:
        api_key: Optional API key (uses OPENAI_API_KEY env var if not provided)
    
    Returns:
        OpenAI client instance
    """
    try:
        from openai import OpenAI
        
        # Only pass the API key and nothing else
        if api_key:
            client = OpenAI(api_key=api_key)
        else:
            client = OpenAI()
            
        logger.info("Successfully created OpenAI client")
        return client
    except Exception as e:
        logger.error(f"Error creating OpenAI client: {str(e)}")
        raise