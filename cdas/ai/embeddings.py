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
    """Manages document embeddings for semantic search.
    
    This class generates and manages vector embeddings for documents, enabling
    semantic search capabilities within the construction document database.
    It uses OpenAI's embedding models to convert text into high-dimensional
    vector representations for similarity search.
    
    Attributes:
        db_session: Database session for accessing and storing embeddings
        config (Dict[str, Any]): Configuration settings for the embedding manager
        mock_mode (bool): Whether mock mode is enabled (no actual API calls)
        embedding_model (str): The embedding model to use (e.g., 'text-embedding-3-small')
        api_key (str): OpenAI API key for authentication
        client: The OpenAI client instance
    
    Examples:
        >>> from cdas.db.session import get_session
        >>> from cdas.ai.embeddings import EmbeddingManager
        >>> 
        >>> session = get_session()
        >>> 
        >>> # Initialize embedding manager
        >>> embedding_manager = EmbeddingManager(session)
        >>> 
        >>> # Generate embeddings for a document
        >>> doc_id = "doc_123abc"
        >>> embedding_manager.embed_document(doc_id)
        >>> 
        >>> # Search for similar documents
        >>> results = embedding_manager.search("HVAC installation costs")
        >>> for result in results:
        ...     print(f"Document: {result['doc_id']}, Similarity: {result['similarity']:.2f}")
    """
    
    def __init__(self, db_session, config: Optional[Dict[str, Any]] = None):
        """Initialize the embedding manager.
        
        Args:
            db_session: Database session for accessing and storing embeddings
            config (Optional[Dict[str, Any]]): Configuration dictionary with the following options:
                - embedding_model (str): Embedding model to use (default: 'text-embedding-3-small')
                - api_key (str): OpenAI API key for authentication
                - mock_mode (bool): Force mock mode (no actual API calls)
        
        Raises:
            ImportError: If the OpenAI package is not installed
            Exception: If there's an error initializing the OpenAI client (will fall back to mock mode)
        
        Note:
            In mock mode, the manager generates deterministic mock embeddings based on
            the hash of the input text, instead of making API calls.
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
        
        This method converts text into a vector embedding using OpenAI's embedding models.
        The resulting embedding can be used for semantic similarity search.
        In mock mode, it generates deterministic mock embeddings based on a hash of the input text.
        
        Args:
            text (str): Text content to convert into an embedding vector
            
        Returns:
            List[float]: Vector embedding representing the semantic content of the text
                (1536 dimensions for text-embedding-3-small, 3072 for text-embedding-3-large)
            
        Raises:
            ValueError: If the embedding client is not initialized
            Exception: If there's an error in the API call (will fall back to mock mode)
            
        Note:
            The vector dimensionality depends on the embedding model:
            - text-embedding-3-small: 1536 dimensions
            - text-embedding-3-large: 3072 dimensions
            
        Examples:
            >>> embedding_manager = EmbeddingManager(session)
            >>> text = "HVAC installation and commissioning"
            >>> embedding = embedding_manager.generate_embeddings(text)
            >>> print(f"Embedding dimensions: {len(embedding)}")
            Embedding dimensions: 1536
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
        
        This method retrieves the content of each page in a document, generates embeddings
        for each page's content, and stores these embeddings in the database. This enables
        semantic search functionality for the document.
        
        Args:
            doc_id (str): Document ID to generate embeddings for
            
        Returns:
            bool: True if embeddings were successfully generated and stored, False otherwise
            
        Raises:
            Exception: If an error occurs during embedding generation or database operations,
                the exception is logged and re-raised
                
        Note:
            Embeddings are stored at the page level, not the document level. Each page in the
            document will have its own embedding vector stored in the pages table, allowing
            for fine-grained semantic search within documents.
            
        Examples:
            >>> embedding_manager = EmbeddingManager(session)
            >>> doc_id = "doc_123abc"
            >>> success = embedding_manager.embed_document(doc_id)
            >>> print(f"Embedding successful: {success}")
            Embedding successful: True
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
        """Search for documents similar to query using semantic search.
        
        This method performs semantic similarity search across document pages by converting
        the query text into an embedding vector and comparing it with stored document
        embeddings. It returns the most semantically similar document pages sorted by
        similarity score.
        
        Args:
            query (str): Search query text
            limit (int, optional): Maximum number of results to return (default: 5)
            
        Returns:
            List[Dict[str, Any]]: List of matching document pages with their metadata and similarity scores.
                Each dictionary contains:
                - page_id: Unique identifier for the page
                - doc_id: Document identifier
                - page_number: Page number within the document
                - content: Page content
                - doc_type: Type of document (e.g., 'payment_app', 'change_order')
                - party: Party associated with the document (e.g., 'contractor', 'district')
                - similarity: Similarity score (0-1, higher is more similar)
            
        Raises:
            Exception: If an error occurs during embedding generation or database operations,
                the exception is logged and re-raised
                
        Note:
            Similarity scores range from 0 to 1, with 1 being an exact match.
            The PostgreSQL operator <=> is used for vector similarity calculation.
            
        Examples:
            >>> embedding_manager = EmbeddingManager(session)
            >>> results = embedding_manager.search("HVAC installation costs", limit=3)
            >>> for result in results:
            ...     print(f"Doc: {result['doc_id']}, Page: {result['page_number']}, "
            ...           f"Similarity: {result['similarity']:.2f}")
            Doc: doc_456def, Page: 2, Similarity: 0.89
            Doc: doc_123abc, Page: 5, Similarity: 0.78
            Doc: doc_789ghi, Page: 1, Similarity: 0.65
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
