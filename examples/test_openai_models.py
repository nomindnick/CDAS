#!/usr/bin/env python3
"""
Simple test script to verify OpenAI API access and list available models.
"""
import os
import sys
from pathlib import Path
import logging

# Add the project root to Python path to ensure imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the openai wrapper from CDAS
from cdas.ai.openai_wrapper import create_openai_client, apply_openai_patches

def test_openai_models():
    """Test the OpenAI API connection and list available models."""
    logger.info("Testing OpenAI API connection and listing models...")
    
    # Apply the monkey patch for the 'proxies' parameter issue
    success = apply_openai_patches()
    if not success:
        logger.warning("Could not apply OpenAI patches, attempting anyway")
    
    # Get the API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return False
    
    logger.info(f"Found API key starting with: {api_key[:10]}...")
    
    try:
        # Create the OpenAI client
        client = create_openai_client(api_key=api_key)
        
        # List available models
        logger.info("Retrieving list of available models...")
        models = client.models.list()
        
        # Log the models
        logger.info(f"Successfully retrieved {len(models.data)} models:")
        for model in models.data:
            logger.info(f"- {model.id}")
        
        # Check if o4-mini is available
        o4_mini_available = any(model.id == "o4-mini" for model in models.data)
        logger.info(f"o4-mini model available: {o4_mini_available}")
        
        # If we got here, everything worked
        logger.info("✅ API test successful!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing OpenAI API: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_openai_models()
    sys.exit(0 if success else 1)