"""Embedding management module.

This module provides functionality for generating and managing document embeddings
for semantic search capabilities.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union

from cdas.ai.monkey_patch import check_mock_mode_config

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages document embeddings for semantic search."""
    
    def __init__(self, db_session, config: Optional[Dict[str, Any]] = None):
        """Initialize the embedding manager.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Check if mock mode is enabled from environment or config
        self.mock_mode = check_mock_mode_config(self.config)
        logger.info(f"Embedding manager mock mode from config check: {self.mock_mode}")
        
        # Initialize embedding model
        self.embedding_model = self.config.get('embedding_model', 'text-embedding-3-small')  # New OpenAI embedding model
        self.api_key = self.config.get('api_key')
        self.client = None
        
        if self.mock_mode:
            logger.warning(
                "Initializing EmbeddingManager in MOCK MODE. "
                "API calls will be simulated and no actual requests will be made."
            )
        else:
            try:
                # Use our custom wrapper to avoid the proxies issue
                from cdas.ai.openai_wrapper import create_openai_client
                self.client = create_openai_client(api_key=self.api_key)
                logger.info(f"Initialized OpenAI client for embeddings with model {self.embedding_model}")
            except ImportError:
                logger.error("OpenAI package not installed. Please install with: pip install openai")
                raise
            except Exception as e:
                logger.error(f"Error initializing OpenAI client for embeddings: {str(e)}")
                logger.warning("Falling back to MOCK MODE due to initialization error")
                self.mock_mode = True
    
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Check if we're in mock mode
        if self.mock_mode:
            logger.info("Running in MOCK MODE - generating simulated embeddings")
            
            # Generate fixed-length mock embedding
            # Using a simple hash-based approach to ensure consistent vectors for the same text
            import numpy as np
            import hashlib
            
            # Create a hash of the text
            text_hash = hashlib.md5(text.encode()).hexdigest()
            # Use just the first 8 chars of hash to stay within numpy's seed range (0 to 2^32-1)
            hash_int = int(text_hash[:8], 16)
            
            # Use the hash to seed random number generator for reproducibility
            np.random.seed(hash_int)
            
            # Get the dimensions based on the embedding model
            # text-embedding-3-small is 1536 dimensions
            # text-embedding-3-large is 3072 dimensions
            dimensions = 1536 if self.embedding_model == 'text-embedding-3-small' else 3072
            mock_embedding = list(np.random.rand(dimensions) - 0.5)  # Center around 0
            
            # Normalize the vector to unit length (common for embeddings)
            norm = np.linalg.norm(mock_embedding)
            if norm > 0:
                mock_embedding = [x / norm for x in mock_embedding]
                
            logger.info(f"Generated mock embedding with {dimensions} dimensions")
            return mock_embedding
        
        # Real API usage        
        if not self.client:
            raise ValueError("Embedding client not initialized")
            
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Fall back to mock mode if API call fails
            logger.warning("Falling back to MOCK MODE due to API error")
            self.mock_mode = True
            # Recursive call - now will use the mock mode
            return self.generate_embeddings(text)
    
    def embed_document(self, doc_id: str) -> bool:
        """Generate and store embeddings for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Success flag
        """
        # Get document content
        query = """
            SELECT 
                p.page_id, 
                p.content
            FROM 
                pages p
            WHERE 
                p.doc_id = %s
            ORDER BY 
                p.page_number
        """
        
        try:
            pages = self.db_session.execute(query, (doc_id,)).fetchall()
            
            for page_id, content in pages:
                # Skip if no content
                if not content:
                    continue
                
                # Generate embedding
                embedding = self.generate_embeddings(content)
                
                # Store embedding in database
                update_query = """
                    UPDATE 
                        pages
                    SET 
                        embedding = %s
                    WHERE 
                        page_id = %s
                """
                
                self.db_session.execute(update_query, (embedding, page_id))
            
            # Commit changes
            self.db_session.commit()
            logger.info(f"Successfully embedded document {doc_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error embedding document {doc_id}: {str(e)}")
            self.db_session.rollback()
            raise
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for documents similar to query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of similar documents
        """
        try:
            # Generate embedding for query
            query_embedding = self.generate_embeddings(query)
            
            # Find similar documents
            search_query = """
                SELECT 
                    p.page_id,
                    p.doc_id,
                    p.page_number,
                    p.content,
                    d.doc_type,
                    d.party,
                    1 - (p.embedding <=> %s) AS similarity
                FROM 
                    pages p
                JOIN
                    documents d ON p.doc_id = d.doc_id
                WHERE 
                    p.embedding IS NOT NULL
                ORDER BY 
                    similarity DESC
                LIMIT %s
            """
            
            results = self.db_session.execute(search_query, (query_embedding, limit)).fetchall()
            
            return [dict(zip(
                ['page_id', 'doc_id', 'page_number', 'content', 'doc_type', 'party', 'similarity'],
                result
            )) for result in results]
        except Exception as e:
            logger.error(f"Error searching with embeddings: {str(e)}")
            raise
