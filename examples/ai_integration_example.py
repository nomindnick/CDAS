"""
AI Integration Example for CDAS

This script demonstrates how to use the AI integration features of CDAS,
including the LLMManager, EmbeddingManager, and InvestigatorAgent with
Anthropic's Claude API.

Usage:
    python examples/ai_integration_example.py
"""

import os
import logging
import time
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import CDAS modules
from cdas.ai.llm import LLMManager
from cdas.ai.embeddings import EmbeddingManager
from cdas.ai.agents.investigator import InvestigatorAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Still needed for embeddings

def setup_mock_db_session():
    """Create a mock database session for example purposes.
    This is used if a real database is not available.
    """
    class MockSession:
        def execute(self, query, params=None):
            class MockResult:
                def fetchall(self):
                    # Return sample data based on the query
                    if "documents" in query and "pages" in query:
                        # Mock document search results
                        if params and "billing" in str(params):
                            return [
                                ("doc_123", "Payment Application #3", "payment_app", "contractor", 
                                 time.strptime("2023-05-15", "%Y-%m-%d"), "approved", "page_1", 1,
                                 "This payment application includes billing for foundation work and electrical installations."),
                                ("doc_456", "Change Order #CS-103", "change_order", "contractor", 
                                 time.strptime("2023-04-10", "%Y-%m-%d"), "rejected", "page_1", 1,
                                 "This change order requests additional payment for electrical work that appears to be in the original scope.")
                            ]
                    elif "line_items" in query:
                        # Mock line item search results
                        if params and (any("electrical" in str(p) for p in params) if params else False):
                            return [
                                ("item_789", "doc_123", "Payment Application #3", "payment_app", "contractor", 
                                 time.strptime("2023-05-15", "%Y-%m-%d"), "Electrical work - main building", 
                                 12500.00, 1, 12500.00),
                                ("item_987", "doc_456", "Change Order #CS-103", "change_order", "contractor", 
                                 time.strptime("2023-04-10", "%Y-%m-%d"), "Additional electrical work", 
                                 12500.00, 1, 12500.00)
                            ]
                    
                    # Default empty result
                    return []
            
            return MockResult()
        
        def commit(self):
            # Mock commit
            pass
        
        def rollback(self):
            # Mock rollback
            pass
    
    return MockSession()

def main():
    """Run the AI integration example."""
    # Check if API keys are set
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY environment variable is not set. Running in mock mode.")
        llm_mock_mode = True
    else:
        logger.info(f"Using Anthropic API key: {ANTHROPIC_API_KEY[:5]}...{ANTHROPIC_API_KEY[-4:] if len(ANTHROPIC_API_KEY) > 8 else '****'}")
        # We'll let the components auto-detect if the key is valid and fall back to mock mode if needed
        llm_mock_mode = False
        
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY environment variable is not set. Running embeddings in mock mode.")
        embedding_mock_mode = True
    else:
        logger.info(f"Using OpenAI API key for embeddings: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 8 else '****'}")
        embedding_mock_mode = False
    
    # Check if we should use a real or mock database
    use_real_db = False
    db_path = os.path.expanduser("~/.cdas/cdas.db")
    
    if use_real_db:
        # Create real database connection
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker(bind=engine)
        session = Session()
        logger.info(f"Using real database at {db_path}")
    else:
        # Use mock session
        session = setup_mock_db_session()
        logger.info("Using mock database session")

    try:
        # Initialize LLM Manager with configuration
        llm_config = {
            'provider': 'anthropic',  # Switch to Anthropic
            'model': 'claude-3-7-sonnet-20250219',  # Use Claude model
            'api_key': ANTHROPIC_API_KEY,
            'mock_mode': llm_mock_mode
        }
        
        llm_manager = LLMManager(llm_config)
        
        # Test LLM Manager
        logger.info("Testing LLM Manager with Anthropic Claude...")
        prompt = "Explain in 3-4 sentences how construction change orders can lead to disputes."
        
        # The component handles mock mode internally now
        response = llm_manager.generate(
            prompt=prompt,
            system_prompt="You are an expert in construction law and finance."
        )
        logger.info(f"Claude Response: {response}")
        
        # Initialize Embedding Manager (still using OpenAI)
        try:
            embedding_config = {
                'embedding_model': 'text-embedding-3-small',
                'api_key': OPENAI_API_KEY,
                'mock_mode': embedding_mock_mode
            }
            
            embedding_manager = EmbeddingManager(session, embedding_config)
            
            # Test Embedding Manager
            logger.info("Testing Embedding Manager...")
            text = "Construction change order requesting additional payment for foundation work."
            
            # The component handles mock mode internally now
            embeddings = embedding_manager.generate_embeddings(text)
            logger.info(f"Generated embedding vector with {len(embeddings)} dimensions")
        except Exception as e:
            logger.error(f"Error with Embedding Manager: {str(e)}")
        
        # Initialize Investigator Agent
        try:
            # Configure agent (the agent uses the LLM manager which already has mock mode configured)
            agent_config = {
                'max_iterations': 5
            }
            
            investigator = InvestigatorAgent(session, llm_manager, agent_config)
            
            # Test Investigator Agent
            logger.info("Testing Investigator Agent with Claude...")
            question = "What evidence exists for potential double-billing in this project?"
            
            # Note: This will return mock results if in mock mode
            # This is controlled by the mock mode in the LLM manager
            investigation = investigator.investigate(question)
            
            # If investigation returned results (real or mock)
            if investigation:
                logger.info(f"Investigation steps: {len(investigation.get('investigation_steps', []))}")
                if 'final_report' in investigation and investigation['final_report']:
                    logger.info(f"Final report summary:\n{investigation['final_report'][:200]}...")
                else:
                    logger.info("No final report generated")
            else:
                logger.warning("Investigation returned no results")
        except Exception as e:
            logger.error(f"Error with Investigator Agent: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in AI integration example: {str(e)}")
    finally:
        if use_real_db:
            session.close()

if __name__ == "__main__":
    main()