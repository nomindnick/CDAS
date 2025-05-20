#!/usr/bin/env python3
"""
Script to verify API keys are properly loaded.
Tests both OpenAI and Anthropic connections.
"""
import os
import logging
from dotenv import load_dotenv
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_openai():
    """Verify OpenAI API key and connection."""
    try:
        from openai import OpenAI
        
        # Check if API key is in environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            return False
        
        # Print key info for debugging
        logger.info(f"Found OpenAI API key starting with: {api_key[:4]}...")
        
        # Check if API key format looks valid
        if not api_key.startswith("sk-"):
            logger.error(f"OpenAI API key format doesn't look valid (doesn't start with 'sk-')")
            return False
        
        # Create client from the specified API key
        try:
            client = OpenAI(api_key=api_key)
            logger.info("Successfully created OpenAI client")
            
            # Try a simple API request
            logger.info("Testing OpenAI API connection by listing models...")
            models = client.models.list()
            
            logger.info(f"Successfully retrieved {len(models.data)} models")
            logger.info("First few models:")
            for model in models.data[:5]:
                logger.info(f"- {model.id}")
            
            logger.info("✅ OpenAI API test successful!")
            return True
        except Exception as e:
            logger.error(f"Error testing OpenAI API: {str(e)}")
            return False
    except ImportError:
        logger.error("OpenAI package not installed. Install with: pip install openai")
        return False

def verify_anthropic():
    """Verify Anthropic API key and connection."""
    try:
        try:
            import anthropic
        except ImportError:
            logger.error("Anthropic package not installed. Install with: pip install anthropic")
            return False
            
        # Check if API key is in environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not found in environment variables")
            return False
        
        # Print key info for debugging
        logger.info(f"Found Anthropic API key starting with: {api_key[:4]}...")
        
        # Create client from our wrapper to handle any issues with proxies
        try:
            # First try to use our custom wrapper
            try:
                from cdas.ai.anthropic_wrapper import create_anthropic_client
                client = create_anthropic_client(api_key=api_key)
                logger.info("Successfully created Anthropic client with custom wrapper")
            except ImportError:
                # Fall back to direct client creation
                client = anthropic.Anthropic(api_key=api_key)
                logger.info("Successfully created Anthropic client directly")
            
            # Try a simple API request
            logger.info("Testing Anthropic API connection...")
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=10,
                messages=[
                    {"role": "user", "content": "Hello Claude, are you there?"}
                ]
            )
            
            logger.info(f"Successfully received response: {response.content[0].text[:30]}...")
            logger.info("✅ Anthropic API test successful!")
            return True
        except Exception as e:
            logger.error(f"Error testing Anthropic API: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Error in Anthropic verification: {str(e)}")
        return False

def main():
    """Verify API keys."""
    parser = argparse.ArgumentParser(description='Verify API keys for AI services')
    parser.add_argument('--openai', action='store_true', help='Test OpenAI API key')
    parser.add_argument('--anthropic', action='store_true', help='Test Anthropic API key')
    parser.add_argument('--all', action='store_true', help='Test all API keys')
    
    args = parser.parse_args()
    
    # If no specific arguments, test all
    if not (args.openai or args.anthropic or args.all):
        args.all = True
    
    # Explicitly load environment variables from .env
    load_dotenv(verbose=True)
    
    # Keep track of successes
    success = True
    
    # Test OpenAI if requested
    if args.openai or args.all:
        openai_success = verify_openai()
        success = success and openai_success
    
    # Test Anthropic if requested
    if args.anthropic or args.all:
        anthropic_success = verify_anthropic()
        success = success and anthropic_success
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)