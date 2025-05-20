"""Semantic search implementation.

This module provides semantic search capabilities for finding relevant
documents based on meaning rather than just keywords.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def semantic_search(db_session, embedding_manager, query: str, limit: int = 5, 
                    doc_type: Optional[str] = None, party: Optional[str] = None,
                    min_date: Optional[str] = None, max_date: Optional[str] = None,
                    threshold: float = 0.0, metadata_filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Search for semantically similar content.
    
    Args:
        db_session: Database session
        embedding_manager: Embedding manager
        query: Search query
        limit: Maximum number of results
        doc_type: Optional document type filter
        party: Optional party filter
        min_date: Optional minimum date filter
        max_date: Optional maximum date filter
        threshold: Minimum similarity threshold
        metadata_filters: Additional metadata filters
        
    Returns:
        List of semantically similar documents
    """
    try:
        # Generate embedding for query
        query_embedding = embedding_manager.generate_embeddings(query)
        
        # Build search query
        search_query = """
            SELECT 
                p.page_id,
                p.doc_id,
                p.page_number,
                p.content,
                d.title,
                d.doc_type,
                d.party,
                d.date,
                d.status,
                1 - (p.embedding <=> %s) AS similarity
            FROM 
                pages p
            JOIN
                documents d ON p.doc_id = d.doc_id
            WHERE 
                p.embedding IS NOT NULL
        """
        
        params = [query_embedding]
        
        # Apply standard filters
        if doc_type:
            if isinstance(doc_type, list):
                placeholders = ", ".join(["%s"] * len(doc_type))
                search_query += f" AND d.doc_type IN ({placeholders})"
                params.extend(doc_type)
            else:
                search_query += " AND d.doc_type = %s"
                params.append(doc_type)
        
        if party:
            if isinstance(party, list):
                placeholders = ", ".join(["%s"] * len(party))
                search_query += f" AND d.party IN ({placeholders})"
                params.extend(party)
            else:
                search_query += " AND d.party = %s"
                params.append(party)
        
        # Apply date range filters if provided
        if min_date:
            search_query += " AND d.date >= %s"
            params.append(min_date)
            
        if max_date:
            search_query += " AND d.date <= %s"
            params.append(max_date)
        
        # Apply similarity threshold
        if threshold > 0:
            search_query += " AND (1 - (p.embedding <=> %s)) >= %s"
            params.append(query_embedding)
            params.append(threshold)
        
        # Apply additional metadata filters if provided
        if metadata_filters:
            for key, value in metadata_filters.items():
                if key == 'status':
                    search_query += " AND d.status = %s"
                    params.append(value)
                elif key == 'file_type':
                    search_query += " AND d.file_type = %s"
                    params.append(value)
                elif key == 'keywords':
                    search_query += " AND p.content ILIKE %s"
                    params.append(f"%{value}%")
                elif key == 'amount_min':
                    search_query += """
                        AND d.doc_id IN (
                            SELECT DISTINCT doc_id FROM line_items WHERE amount >= %s
                        )
                    """
                    params.append(value)
                elif key == 'amount_max':
                    search_query += """
                        AND d.doc_id IN (
                            SELECT DISTINCT doc_id FROM line_items WHERE amount <= %s
                        )
                    """
                    params.append(value)
        
        search_query += """
            ORDER BY 
                similarity DESC
            LIMIT %s
        """
        params.append(limit)
        
        # Execute query
        results = db_session.execute(search_query, params).fetchall()
        
        # Format results
        search_results = []
        for item in results:
            page_id, doc_id, page_number, content, title, doc_type, party, date, status, similarity = item
            
            # Extract context (up to 300 chars around most relevant content)
            context = content[:500] + "..." if len(content) > 500 else content
            
            search_results.append({
                'page_id': page_id,
                'doc_id': doc_id,
                'page_number': page_number,
                'document': {
                    'title': title,
                    'doc_type': doc_type,
                    'party': party,
                    'date': date.isoformat() if date else None,
                    'status': item[8]  # Status from the query
                },
                'context': context,
                'similarity': similarity,
                'match_score': round(similarity * 100)  # Percentage score for easier interpretation
            })
        
        return search_results
        
    except Exception as e:
        logger.error(f"Error performing semantic search: {str(e)}")
        raise


def batch_embed_documents(db_session, embedding_manager, doc_ids: Optional[List[str]] = None, 
                          doc_type: Optional[str] = None, party: Optional[str] = None,
                          min_date: Optional[str] = None, max_date: Optional[str] = None,
                          use_vectorizer: bool = False, limit: Optional[int] = None) -> Dict[str, Any]:
    """Batch embed documents.
    
    Args:
        db_session: Database session
        embedding_manager: Embedding manager
        doc_ids: Optional list of document IDs to embed
        doc_type: Optional document type filter
        party: Optional party filter
        min_date: Optional minimum date filter
        max_date: Optional maximum date filter
        use_vectorizer: Whether to use vectorizer for better chunking and preprocessing
        limit: Optional maximum number of documents to embed
        
    Returns:
        Embedding results
    """
    try:
        # Find documents to embed
        if doc_ids:
            query = """SELECT doc_id FROM documents WHERE doc_id IN %s"""
            results = db_session.execute(query, (tuple(doc_ids),)).fetchall()
            documents = [row[0] for row in results]
        else:
            query = """SELECT doc_id FROM documents WHERE 1=1"""
            params = []
            
            if doc_type:
                query += " AND doc_type = %s"
                params.append(doc_type)
            
            if party:
                query += " AND party = %s"
                params.append(party)
                
            if min_date:
                query += " AND date >= %s"
                params.append(min_date)
                
            if max_date:
                query += " AND date <= %s"
                params.append(max_date)
            
            if limit is not None:
                query += " LIMIT %s"
                params.append(limit)
            
            results = db_session.execute(query, params).fetchall()
            documents = [row[0] for row in results]
        
        # Embed documents
        embedded_docs = []
        failed_docs = []
        
        # Use vectorizer if specified for better chunking and preprocessing
        if use_vectorizer:
            try:
                # Import vectorizer here to avoid circular import
                from cdas.ai.semantic_search.vectorizer import Vectorizer
                vectorizer = Vectorizer(embedding_manager)
                
                for doc_id in documents:
                    try:
                        # Get document content and metadata
                        doc_query = """
                            SELECT 
                                d.doc_id, 
                                d.title, 
                                d.doc_type, 
                                d.party, 
                                d.date,
                                d.status,
                                string_agg(p.content, '\n\n' ORDER BY p.page_number) AS content
                            FROM 
                                documents d
                            LEFT JOIN
                                pages p ON d.doc_id = p.doc_id
                            WHERE 
                                d.doc_id = %s
                            GROUP BY
                                d.doc_id
                        """
                        
                        doc_info = db_session.execute(doc_query, (doc_id,)).fetchone()
                        
                        if not doc_info or not doc_info[6]:  # No content
                            failed_docs.append(doc_id)
                            continue
                        
                        # Prepare document for vectorizer
                        document = {
                            'doc_id': doc_info[0],
                            'title': doc_info[1],
                            'doc_type': doc_info[2],
                            'party': doc_info[3],
                            'date': doc_info[4].isoformat() if doc_info[4] else None,
                            'status': doc_info[5],
                            'content': doc_info[6]
                        }
                        
                        # Vectorize document into chunks with embeddings
                        vectorized_chunks = vectorizer.vectorize_document(document)
                        
                        if not vectorized_chunks:
                            failed_docs.append(doc_id)
                            continue
                        
                        # Store embeddings for each chunk
                        for i, chunk in enumerate(vectorized_chunks):
                            # Create a new page for each chunk
                            page_query = """
                                INSERT INTO pages 
                                    (doc_id, page_number, content, embedding) 
                                VALUES 
                                    (%s, %s, %s, %s)
                                ON CONFLICT (doc_id, page_number) 
                                DO UPDATE SET 
                                    content = EXCLUDED.content, 
                                    embedding = EXCLUDED.embedding
                            """
                            
                            db_session.execute(page_query, (
                                doc_id, 
                                i + 1,  # Page number starts at 1
                                chunk['text'], 
                                chunk['embedding']
                            ))
                        
                        embedded_docs.append(doc_id)
                    except Exception as e:
                        logger.error(f"Error vectorizing document {doc_id}: {str(e)}")
                        failed_docs.append(doc_id)
            except ImportError:
                logger.warning("Vectorizer not available, falling back to standard embedding")
                use_vectorizer = False
        
        # Use standard embedding if not using vectorizer
        if not use_vectorizer:
            for doc_id in documents:
                try:
                    success = embedding_manager.embed_document(doc_id)
                    if success:
                        embedded_docs.append(doc_id)
                    else:
                        failed_docs.append(doc_id)
                except Exception as e:
                    logger.error(f"Error embedding document {doc_id}: {str(e)}")
                    failed_docs.append(doc_id)
        
        return {
            'total_documents': len(documents),
            'embedded_documents': len(embedded_docs),
            'failed_documents': len(failed_docs),
            'embedded_doc_ids': embedded_docs,
            'failed_doc_ids': failed_docs
        }
        
    except Exception as e:
        logger.error(f"Error batch embedding documents: {str(e)}")
        raise


def semantic_query(db_session, embedding_manager, query: str, filters: Optional[Dict[str, Any]] = None,
                  use_hybrid: bool = True, limit: int = 10) -> List[Dict[str, Any]]:
    """Perform a hybrid semantic and keyword search.
    
    Args:
        db_session: Database session
        embedding_manager: Embedding manager
        query: Search query text
        filters: Optional filters to apply to search
        use_hybrid: Whether to use hybrid search (semantic + keyword)
        limit: Maximum number of results
        
    Returns:
        List of search results
    """
    try:
        # Process filters
        filters = filters or {}
        doc_type = filters.get('doc_type')
        party = filters.get('party')
        min_date = filters.get('min_date')
        max_date = filters.get('max_date')
        threshold = filters.get('threshold', 0.0)
        metadata_filters = {k: v for k, v in filters.items() 
                          if k not in ['doc_type', 'party', 'min_date', 'max_date', 'threshold']}
        
        # Perform semantic search
        semantic_results = semantic_search(
            db_session, 
            embedding_manager, 
            query, 
            limit=limit * 2 if use_hybrid else limit,  # Get more results if doing hybrid search
            doc_type=doc_type,
            party=party,
            min_date=min_date,
            max_date=max_date,
            threshold=threshold,
            metadata_filters=metadata_filters
        )
        
        # If not using hybrid search, return semantic results directly
        if not use_hybrid:
            return semantic_results[:limit]
        
        # Perform keyword search for hybrid approach
        keyword_query = """
            SELECT 
                p.page_id,
                p.doc_id,
                p.page_number,
                p.content,
                d.title,
                d.doc_type,
                d.party,
                d.date,
                d.status,
                ts_rank_cd(to_tsvector('english', p.content), plainto_tsquery('english', %s)) AS rank
            FROM 
                pages p
            JOIN
                documents d ON p.doc_id = d.doc_id
            WHERE 
                to_tsvector('english', p.content) @@ plainto_tsquery('english', %s)
        """
        
        params = [query, query]
        
        # Apply the same filters as semantic search
        if doc_type:
            if isinstance(doc_type, list):
                placeholders = ", ".join(["%s"] * len(doc_type))
                keyword_query += f" AND d.doc_type IN ({placeholders})"
                params.extend(doc_type)
            else:
                keyword_query += " AND d.doc_type = %s"
                params.append(doc_type)
        
        if party:
            if isinstance(party, list):
                placeholders = ", ".join(["%s"] * len(party))
                keyword_query += f" AND d.party IN ({placeholders})"
                params.extend(party)
            else:
                keyword_query += " AND d.party = %s"
                params.append(party)
        
        if min_date:
            keyword_query += " AND d.date >= %s"
            params.append(min_date)
            
        if max_date:
            keyword_query += " AND d.date <= %s"
            params.append(max_date)
        
        # Apply additional metadata filters
        if metadata_filters:
            for key, value in metadata_filters.items():
                if key == 'status':
                    keyword_query += " AND d.status = %s"
                    params.append(value)
                elif key == 'file_type':
                    keyword_query += " AND d.file_type = %s"
                    params.append(value)
        
        keyword_query += """
            ORDER BY rank DESC
            LIMIT %s
        """
        params.append(limit)
        
        keyword_results = db_session.execute(keyword_query, params).fetchall()
        
        # Format keyword results
        formatted_keyword_results = []
        for item in keyword_results:
            # Extract context (up to 300 chars around most relevant parts)
            context = item[3][:500] + "..." if len(item[3]) > 500 else item[3]
            
            formatted_keyword_results.append({
                'page_id': item[0],
                'doc_id': item[1],
                'page_number': item[2],
                'document': {
                    'title': item[4],
                    'doc_type': item[5],
                    'party': item[6],
                    'date': item[7].isoformat() if item[7] else None,
                    'status': item[8]
                },
                'context': context,
                'rank': float(item[9]),
                'match_score': round(float(item[9]) * 50),  # Convert rank to comparable score
                'match_type': 'keyword'
            })
        
        # Add match type to semantic results
        for result in semantic_results:
            result['match_type'] = 'semantic'
        
        # Combine and deduplicate results
        combined_results = []
        seen_page_ids = set()
        
        # Process semantic results first (usually higher quality)
        for result in semantic_results:
            page_id = result['page_id']
            if page_id not in seen_page_ids:
                seen_page_ids.add(page_id)
                combined_results.append(result)
        
        # Add keyword results if not already included
        for result in formatted_keyword_results:
            page_id = result['page_id']
            if page_id not in seen_page_ids and len(combined_results) < limit:
                seen_page_ids.add(page_id)
                combined_results.append(result)
        
        # Sort by match score (descending)
        combined_results.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        # Limit to requested number
        return combined_results[:limit]
        
    except Exception as e:
        logger.error(f"Error performing semantic query: {str(e)}")
        raise