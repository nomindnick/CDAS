#!/usr/bin/env python3
"""
Test script for EmbeddingManager with mock mode.
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to Python path to ensure imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import CDAS modules
from cdas.ai.embeddings import EmbeddingManager

class MockSession:
    """Mock database session for testing."""
    
    def execute(self, query, params=None):
        """Mock execute method."""
        logger.info(f"Mock DB execute called with query: {query[:50]}...")
        return self
        
    def fetchall(self):
        """Mock fetchall method."""
        logger.info("Mock DB fetchall called")
        return []
        
    def commit(self):
        """Mock commit method."""
        logger.info("Mock DB commit called")
        
    def rollback(self):
        """Mock rollback method."""
        logger.info("Mock DB rollback called")

def main():
    """Test the EmbeddingManager with mock mode."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Get OpenAI API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    logger.info(f"Found API key starting with: {api_key[:10] if api_key else 'None'}...")
    
    # Create mock DB session
    mock_session = MockSession()
    
    # Create EmbeddingManager with mock mode
    logger.info("Creating EmbeddingManager with mock mode enabled...")
    config = {
        "embedding_model": "text-embedding-3-small",
        "api_key": api_key,
        "mock_mode": True  # Enable mock mode
    }
    
    embedding_manager = EmbeddingManager(mock_session, config)
    
    # Test generating embeddings
    logger.info("\nTesting embedding generation...")
    
    # Test with a short text
    short_text = "Contract agreement between school district and general contractor"
    short_embedding = embedding_manager.generate_embeddings(short_text)
    logger.info(f"Generated embedding for short text with dimensions: {len(short_embedding)}")
    
    # Test with a longer text
    long_text = """The contractor shall furnish all labor, materials, equipment and services 
    necessary to perform all work required for the project in strict accordance with the 
    Contract Documents as prepared by the Architect. The project consists of construction 
    of a new two-story elementary school building of approximately 65,000 square feet."""
    long_embedding = embedding_manager.generate_embeddings(long_text)
    logger.info(f"Generated embedding for long text with dimensions: {len(long_embedding)}")
    
    # Verify that same text produces same embeddings (reproducibility)
    repeat_embedding = embedding_manager.generate_embeddings(short_text)
    vectors_match = short_embedding == repeat_embedding
    logger.info(f"Vectors for same text match (should be True): {vectors_match}")
    
    # Test embedding a document
    logger.info("\nTesting document embedding...")
    result = embedding_manager.embed_document("mock_doc_123")
    
    # Test search
    logger.info("\nTesting semantic search...")
    search_results = embedding_manager.search("change order dispute", limit=3)
    logger.info(f"Search returned {len(search_results)} results")
    
    logger.info("\n✅ EmbeddingManager mock mode test completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)