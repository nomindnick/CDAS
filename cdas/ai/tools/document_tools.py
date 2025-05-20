"""Document-related tools for agent use.

This module provides document-related tools that can be used by agents
for document retrieval and analysis.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def search_documents(db_session, query: str, doc_type: Optional[str] = None, 
                    party: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search for documents matching a query.
    
    Args:
        db_session: Database session
        query: Search query
        doc_type: Optional document type filter
        party: Optional party filter
        
    Returns:
        List of matching documents
    """
    try:
        # Build SQL query
        sql_query = """
            SELECT 
                d.doc_id,
                d.title,
                d.doc_type,
                d.party,
                d.date,
                d.status,
                p.page_id,
                p.page_number,
                p.content
            FROM 
                documents d
            JOIN
                pages p ON d.doc_id = p.doc_id
            WHERE 
                1=1
        """
        
        params = []
        
        if query:
            sql_query += " AND p.content ILIKE %s"
            params.append(f"%{query}%")
        
        if doc_type:
            sql_query += " AND d.doc_type = %s"
            params.append(doc_type)
        
        if party:
            sql_query += " AND d.party = %s"
            params.append(party)
        
        sql_query += " ORDER BY d.date DESC, p.page_number ASC"
        
        # Execute query
        results = db_session.execute(sql_query, params).fetchall()
        
        # Format results
        docs = {}
        for doc_id, title, doc_type, party, date, status, page_id, page_number, content in results:
            if doc_id not in docs:
                docs[doc_id] = {
                    'doc_id': doc_id,
                    'title': title,
                    'doc_type': doc_type,
                    'party': party,
                    'date': date.isoformat() if date else None,
                    'status': status,
                    'pages': []
                }
            
            if content and query and query.lower() in content.lower():
                # Extract context around the match
                start_idx = max(0, content.lower().find(query.lower()) - 100)
                end_idx = min(len(content), content.lower().find(query.lower()) + len(query) + 100)
                context = content[start_idx:end_idx]
                
                # Add page with context
                docs[doc_id]['pages'].append({
                    'page_id': page_id,
                    'page_number': page_number,
                    'context': f"...{context}..."
                })
        
        return list(docs.values())
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise


def get_document_content(db_session, doc_id: str) -> Dict[str, Any]:
    """Get the full content of a document.
    
    Args:
        db_session: Database session
        doc_id: Document ID
        
    Returns:
        Document content
    """
    try:
        # Get document metadata
        doc_query = """
            SELECT 
                d.doc_id,
                d.title,
                d.doc_type,
                d.party,
                d.date,
                d.status
            FROM 
                documents d
            WHERE 
                d.doc_id = %s
        """
        
        doc_result = db_session.execute(doc_query, (doc_id,)).fetchone()
        if not doc_result:
            logger.warning(f"Document not found: {doc_id}")
            return {}
        
        doc_id, title, doc_type, party, date, status = doc_result
        
        # Get document pages
        pages_query = """
            SELECT 
                p.page_id,
                p.page_number,
                p.content
            FROM 
                pages p
            WHERE 
                p.doc_id = %s
            ORDER BY 
                p.page_number
        """
        
        pages_result = db_session.execute(pages_query, (doc_id,)).fetchall()
        
        pages = []
        for page_id, page_number, content in pages_result:
            pages.append({
                'page_id': page_id,
                'page_number': page_number,
                'content': content
            })
        
        # Get document line items
        items_query = """
            SELECT 
                li.item_id,
                li.description,
                li.amount,
                li.quantity,
                li.unit_price
            FROM 
                line_items li
            WHERE 
                li.doc_id = %s
            ORDER BY 
                li.item_id
        """
        
        items_result = db_session.execute(items_query, (doc_id,)).fetchall()
        
        line_items = []
        for item_id, description, amount, quantity, unit_price in items_result:
            line_items.append({
                'item_id': item_id,
                'description': description,
                'amount': amount,
                'quantity': quantity,
                'unit_price': unit_price
            })
        
        return {
            'doc_id': doc_id,
            'title': title,
            'doc_type': doc_type,
            'party': party,
            'date': date.isoformat() if date else None,
            'status': status,
            'pages': pages,
            'line_items': line_items
        }
        
    except Exception as e:
        logger.error(f"Error getting document content: {str(e)}")
        raise


def compare_documents(db_session, doc_id1: str, doc_id2: str) -> Dict[str, Any]:
    """Compare two documents.
    
    Args:
        db_session: Database session
        doc_id1: First document ID
        doc_id2: Second document ID
        
    Returns:
        Comparison results
    """
    try:
        # Get document contents
        doc1 = get_document_content(db_session, doc_id1)
        doc2 = get_document_content(db_session, doc_id2)
        
        if not doc1 or not doc2:
            return {'error': 'One or both documents not found'}
        
        # Find common line items (by description similarity and amount)
        common_items = []
        for item1 in doc1.get('line_items', []):
            for item2 in doc2.get('line_items', []):
                # Check for similar description (exact match or contains)
                desc_match = False
                if item1['description'] and item2['description']:
                    desc1 = item1['description'].lower()
                    desc2 = item2['description'].lower()
                    if desc1 == desc2 or desc1 in desc2 or desc2 in desc1:
                        desc_match = True
                
                # Check for matching amount (exact or within 1%)
                amount_match = False
                if item1['amount'] is not None and item2['amount'] is not None:
                    tolerance = max(item1['amount'], item2['amount']) * 0.01
                    if abs(item1['amount'] - item2['amount']) <= tolerance:
                        amount_match = True
                
                if desc_match or amount_match:
                    common_items.append({
                        'item1': item1,
                        'item2': item2,
                        'desc_match': desc_match,
                        'amount_match': amount_match
                    })
        
        # Find differences in amounts for similar items
        amount_differences = []
        for common in common_items:
            if common['desc_match'] and not common['amount_match']:
                amount_differences.append({
                    'description': common['item1']['description'],
                    'amount1': common['item1']['amount'],
                    'amount2': common['item2']['amount'],
                    'difference': common['item2']['amount'] - common['item1']['amount'],
                    'percentage': (common['item2']['amount'] - common['item1']['amount']) / common['item1']['amount'] * 100 if common['item1']['amount'] != 0 else None
                })
        
        return {
            'doc1': {
                'doc_id': doc1['doc_id'],
                'title': doc1['title'],
                'doc_type': doc1['doc_type'],
                'party': doc1['party'],
                'date': doc1['date'],
                'total_items': len(doc1.get('line_items', []))
            },
            'doc2': {
                'doc_id': doc2['doc_id'],
                'title': doc2['title'],
                'doc_type': doc2['doc_type'],
                'party': doc2['party'],
                'date': doc2['date'],
                'total_items': len(doc2.get('line_items', []))
            },
            'common_items': common_items,
            'amount_differences': amount_differences,
            'total_common_items': len(common_items),
            'total_amount_differences': len(amount_differences)
        }
        
    except Exception as e:
        logger.error(f"Error comparing documents: {str(e)}")
        raise
