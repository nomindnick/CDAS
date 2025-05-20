"""Question answering functionality using AI.

This module provides functionality for answering natural language questions
about construction documents and financial data using AI.
"""

import logging
from typing import Dict, List, Optional, Any, Union

from cdas.db.session import session_scope
from cdas.ai.llm import LLMManager
from cdas.ai.embeddings import EmbeddingManager
from cdas.ai.agents.investigator import InvestigatorAgent
from cdas.ai.semantic_search.search import semantic_search
from cdas.config import get_config

logger = logging.getLogger(__name__)


def setup_ai_managers(session) -> tuple:
    """Set up AI managers.
    
    Args:
        session: Database session
        
    Returns:
        Tuple of (llm_manager, embedding_manager)
    """
    config = get_config()
    ai_config = config.get('ai', {})
    
    # Create LLM manager
    llm_manager = LLMManager(ai_config.get('llm', {}))
    
    # Create embedding manager
    embedding_manager = EmbeddingManager(session, ai_config.get('embeddings', {}))
    
    return llm_manager, embedding_manager


def answer_question(question: str) -> Dict[str, Any]:
    """Answer a natural language question about the construction documents.
    
    Args:
        question: Question to answer
        
    Returns:
        Dictionary with answer and supporting information
    """
    logger.info(f"Answering question: {question}")
    
    with session_scope() as session:
        # Set up AI managers
        llm_manager, embedding_manager = setup_ai_managers(session)
        
        # Perform semantic search to find relevant information
        try:
            search_results = semantic_search(
                session, embedding_manager, question, limit=10
            )
            logger.info(f"Found {len(search_results)} relevant documents for question")
        except Exception as e:
            logger.warning(f"Error in semantic search: {str(e)}")
            search_results = []
        
        # Create context from search results
        context = ""
        if search_results:
            context += "Relevant information from documents:\n\n"
            for i, result in enumerate(search_results[:5]):  # Limit to top 5 results
                doc_info = result['document']
                context += f"Document {i+1}: {doc_info['title']} ({doc_info['doc_type']} from {doc_info['party']}, {doc_info['date']})\n"
                context += f"Content: {result['context']}\n\n"
        
        # Create investigator agent
        agent = InvestigatorAgent(session, llm_manager)
        
        # Investigate question
        try:
            investigation_results = agent.investigate(question, context)
            
            return {
                'question': question,
                'answer': investigation_results.get('final_report', 'Unable to generate answer'),
                'investigation_steps': investigation_results.get('investigation_steps', []),
                'search_results': search_results,
                'error': investigation_results.get('error')
            }
        except Exception as e:
            logger.error(f"Error in investigation: {str(e)}")
            
            # Fallback to simple question answering
            prompt = f"""I need to answer a question about a construction dispute based on available documents.

Question: {question}

Context:
{context}

Please provide a concise answer based on the available information. If the answer cannot be determined from the context, explain why and what additional information would be needed.
"""
            
            try:
                answer = llm_manager.generate(prompt)
                return {
                    'question': question,
                    'answer': answer,
                    'search_results': search_results,
                    'error': str(e)
                }
            except Exception as fallback_error:
                logger.error(f"Error in fallback question answering: {str(fallback_error)}")
                return {
                    'question': question,
                    'answer': "Unable to generate an answer due to technical issues.",
                    'error': f"{str(e)}; Fallback error: {str(fallback_error)}"
                }