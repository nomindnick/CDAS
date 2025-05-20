#!/usr/bin/env python3
"""
Simple test script to check real API functionality.
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# Force disable mock mode
if "CDAS_MOCK_MODE" in os.environ:
    del os.environ["CDAS_MOCK_MODE"]

# Use DotEnv to load API keys
from dotenv import load_dotenv
load_dotenv(verbose=True)

# Import our modules
from cdas.ai.llm import LLMManager
from cdas.ai.monkey_patch import is_mock_mode, check_mock_mode_config

def main():
    """Main test function."""
    logger.info("Testing real API functionality")
    
    # Check if mock mode is active
    logger.info(f"Raw check_mock_mode: {is_mock_mode()}")
    logger.info(f"Config check_mock_mode: {check_mock_mode_config()}")
    
    # Create explicit non-mock config
    config = {"provider": "anthropic", "mock_mode": False}
    
    # Create LLM manager
    logger.info("Creating LLM manager with explicit mock_mode=False")
    llm = LLMManager(config)
    
    # Check if mock mode is enabled
    logger.info(f"LLM manager mock mode: {llm.mock_mode}")
    
    # Try a test generation
    logger.info("Testing generation")
    result = llm.generate("What is 2+2? Give just the number.")
    logger.info(f"Response: {result}")
    
    # Check if the result looks like a mock response
    is_mock = "[MOCK]" in result or "MOCK RESPONSE" in result
    logger.info(f"Is mock response: {is_mock}")
    
    return not is_mock  # Return True for success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)