"""
Document relationship analyzer for the financial analysis engine.

This module analyzes relationships between documents based on their content,
metadata, chronology, and financial data.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
from decimal import Decimal

import logging
logger = logging.getLogger(__name__)


class DocumentRelationshipAnalyzer:
    """Analyzes relationships between documents."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the document relationship analyzer.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Default configuration
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.auto_register = self.config.get('auto_register_relationships', True)
        
        # Known relationship types
        self.known_relationship_types = {
            'parent_child': "Parent-child relationship",
            'revision': "Document revision",
            'response': "Response to document",
            'approval': "Approval of document",
            'rejection': "Rejection of document",
            'reference': "Reference to document",
            'amendment': "Amendment to document",
            'contains_same_items': "Contains same line items"
        }
        
    def analyze_document(self, doc_id: str) -> Dict[str, Any]:
        """Analyze relationships for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Relationship analysis
        """
        logger.info(f"Analyzing document relationships for document: {doc_id}")
        
        # Get document information
        document = self._get_document(doc_id)
        if not document:
            logger.warning(f"Document not found: {doc_id}")
            return {'error': 'Document not found'}
        
        # First check existing relationships in database
        existing_relationships = self._get_existing_relationships(doc_id)
        
        # Detect potential relationships that may not be registered yet
        potential_relationships = self._detect_potential_relationships(doc_id, document)
        
        # Combine existing and potential relationships
        # For potential relationships that match existing ones, use the existing data
        all_relationships = existing_relationships.copy()
        
        for potential_rel in potential_relationships:
            # Check if this relationship already exists
            exists = False
            for existing_rel in existing_relationships:
                if (potential_rel['target_doc_id'] == existing_rel['target_doc_id'] and
                    potential_rel['relationship_type'] == existing_rel['relationship_type']):
                    exists = True
                    break
            
            if not exists:
                all_relationships.append(potential_rel)
        
        # Categorize relationships by type
        relationships_by_type = {}
        for rel in all_relationships:
            rel_type = rel['relationship_type']
            if rel_type not in relationships_by_type:
                relationships_by_type[rel_type] = []
            relationships_by_type[rel_type].append(rel)
        
        # Find related documents with shared line items
        shared_item_relationships = self._find_shared_item_relationships(doc_id)
        
        # Group shared item relationships
        if shared_item_relationships:
            relationships_by_type['contains_same_items'] = shared_item_relationships
        
        # Register new relationships if auto-registration is enabled
        if self.auto_register:
            self._register_new_relationships(doc_id, potential_relationships, existing_relationships)
        
        # Return analysis results
        result = {
            'document': document,
            'relationships': all_relationships,
            'relationships_by_type': relationships_by_type,
            'existing_count': len(existing_relationships),
            'potential_count': len(potential_relationships),
            'total_count': len(all_relationships)
        }
        
        logger.info(f"Document relationship analysis complete: {len(all_relationships)} total relationships")
        return result
    
    def analyze_multi_document_relationships(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Analyze relationships between multiple documents.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            Relationship analysis
        """
        logger.info(f"Analyzing relationships between {len(doc_ids)} documents")
        
        # Get details for all documents
        documents = {}
        for doc_id in doc_ids:
            doc = self._get_document(doc_id)
            if doc:
                documents[doc_id] = doc
        
        if not documents:
            logger.warning("No valid documents found")
            return {'error': 'No valid documents found'}
        
        # Get relationships for all documents
        all_relationships = []
        for doc_id in documents:
            # Get existing relationships
            existing_relationships = self._get_existing_relationships(doc_id)
            all_relationships.extend(existing_relationships)
        
        # Get relationships between the specified documents
        cross_relationships = []
        for i, doc_id1 in enumerate(doc_ids):
            for doc_id2 in doc_ids[i+1:]:
                # Skip if either document is invalid
                if doc_id1 not in documents or doc_id2 not in documents:
                    continue
                    
                # Check for relationships between these documents
                cross_rels = self._detect_relationship_between_documents(
                    doc_id1, documents[doc_id1], 
                    doc_id2, documents[doc_id2]
                )
                
                cross_relationships.extend(cross_rels)
        
        # Build relationship graph
        graph = self._build_relationship_graph(documents, all_relationships + cross_relationships)
        
        # Register new cross-document relationships if auto-registration is enabled
        if self.auto_register:
            for rel in cross_relationships:
                # Skip if already in database (would have been in all_relationships)
                is_existing = any(
                    existing_rel['source_doc_id'] == rel['source_doc_id'] and
                    existing_rel['target_doc_id'] == rel['target_doc_id'] and
                    existing_rel['relationship_type'] == rel['relationship_type']
                    for existing_rel in all_relationships
                )
                
                if not is_existing:
                    self._register_relationship(
                        rel['source_doc_id'], 
                        rel['target_doc_id'], 
                        rel['relationship_type'], 
                        rel['confidence']
                    )
        
        # Return analysis results
        result = {
            'documents': list(documents.values()),
            'relationships': all_relationships + cross_relationships,
            'cross_document_relationships': cross_relationships,
            'graph': graph
        }
        
        logger.info(f"Multi-document relationship analysis complete: {len(result['relationships'])} total relationships")
        return result
    
    def find_related_documents(self, doc_id: str, relation_types: Optional[List[str]] = None, 
                          min_confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        """Find documents related to a specific document.
        
        Args:
            doc_id: Document ID
            relation_types: Optional list of relationship types to filter by
            min_confidence: Minimum confidence threshold (defaults to self.min_confidence)
            
        Returns:
            List of related documents
        """
        logger.info(f"Finding related documents for document: {doc_id}")
        
        min_confidence = min_confidence if min_confidence is not None else self.min_confidence
        
        # Build SQL query for document relationships
        query = text("""
            SELECT 
                r.source_doc_id,
                r.target_doc_id,
                r.relationship_type,
                r.confidence,
                d.doc_type,
                d.party,
                d.date_created,
                d.status
            FROM 
                document_relationships r
            JOIN 
                documents d ON (
                    CASE 
                        WHEN r.source_doc_id = :doc_id THEN r.target_doc_id = d.doc_id
                        ELSE r.source_doc_id = d.doc_id
                    END
                )
            WHERE 
                (r.source_doc_id = :doc_id OR r.target_doc_id = :doc_id)
                AND r.confidence >= :min_confidence
                {relation_filter}
            ORDER BY 
                r.confidence DESC
        """)
        
        # Add relation type filter if specified
        relation_filter = ""
        params = {"doc_id": doc_id, "min_confidence": min_confidence}
        
        if relation_types:
            relation_placeholders = [f":rel_type_{i}" for i in range(len(relation_types))]
            relation_filter = f"AND r.relationship_type IN ({', '.join(relation_placeholders)})"
            
            for i, rel_type in enumerate(relation_types):
                params[f"rel_type_{i}"] = rel_type
        
        # Execute query with appropriate parameters
        formatted_query = query.text.replace("{relation_filter}", relation_filter)
        query = text(formatted_query)
        relationships = self.db_session.execute(query, params).fetchall()
        
        # Process results
        related_docs = []
        for rel in relationships:
            source_id, target_id, rel_type, confidence, doc_type, party, date_created, status = rel
            
            # Determine which ID is the related document
            related_id = target_id if source_id == doc_id else source_id
            
            # Determine relationship direction
            direction = "outgoing" if source_id == doc_id else "incoming"
            
            related_docs.append({
                'doc_id': related_id,
                'relationship_type': rel_type,
                'relationship_direction': direction,
                'confidence': float(confidence) if confidence is not None else None,
                'doc_type': doc_type,
                'party': party,
                'date': date_created.isoformat() if date_created else None,
                'status': status
            })
        
        logger.info(f"Found {len(related_docs)} related documents")
        return related_docs
    
    def document_relationship_chain(self, doc_id: str, max_depth: int = 3) -> Dict[str, Any]:
        """Build a chain of related documents.
        
        Args:
            doc_id: Starting document ID
            max_depth: Maximum depth of the relationship chain
            
        Returns:
            Relationship chain
        """
        logger.info(f"Building document relationship chain for document: {doc_id}")
        
        # Get document details
        document = self._get_document(doc_id)
        if not document:
            logger.warning(f"Document not found: {doc_id}")
            return {'error': 'Document not found'}
        
        # Initialize chain with starting document
        chain = {
            'document': document,
            'prev': None,
            'next': None,
            'related': []
        }
        
        # Process relationships iteratively, both forward and backward
        visited = {doc_id}
        
        # Find previous documents in chain
        prev_node = chain
        depth = 0
        while depth < max_depth:
            # Find previous document relationship (most likely parent or predecessor)
            prev_id = self._find_previous_document(prev_node['document']['doc_id'], visited)
            
            if not prev_id:
                break
                
            # Get document details
            prev_document = self._get_document(prev_id)
            if not prev_document:
                break
                
            # Add to chain
            prev_node['prev'] = {
                'document': prev_document,
                'prev': None,
                'next': prev_node,
                'related': []
            }
            
            visited.add(prev_id)
            prev_node = prev_node['prev']
            depth += 1
        
        # Find next documents in chain
        next_node = chain
        depth = 0
        while depth < max_depth:
            # Find next document relationship (most likely child or successor)
            next_id = self._find_next_document(next_node['document']['doc_id'], visited)
            
            if not next_id:
                break
                
            # Get document details
            next_document = self._get_document(next_id)
            if not next_document:
                break
                
            # Add to chain
            next_node['next'] = {
                'document': next_document,
                'prev': next_node,
                'next': None,
                'related': []
            }
            
            visited.add(next_id)
            next_node = next_node['next']
            depth += 1
        
        # Find related documents for each node in the chain
        current = chain
        while current:
            # Find related documents
            related_ids = self._find_related_documents(current['document']['doc_id'], visited)
            
            for related_id in related_ids:
                # Get document details
                related_document = self._get_document(related_id)
                if not related_document:
                    continue
                    
                # Add to related documents
                current['related'].append({
                    'document': related_document
                })
                
                visited.add(related_id)
            
            current = current['next']
        
        logger.info(f"Document relationship chain built with {len(visited)} documents")
        return chain
    
    def register_relationship(self, source_doc_id: str, target_doc_id: str, 
                          relationship_type: str, confidence: float = 1.0) -> Dict[str, Any]:
        """Register a relationship between two documents.
        
        Args:
            source_doc_id: Source document ID
            target_doc_id: Target document ID
            relationship_type: Type of relationship
            confidence: Confidence score for the relationship
            
        Returns:
            Dictionary with relationship information
        """
        return self._register_relationship(source_doc_id, target_doc_id, relationship_type, confidence)
    
    def _get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document details.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document details or None if not found
        """
        query = text("""
            SELECT 
                d.doc_id,
                d.file_name,
                d.doc_type,
                d.party,
                d.date_created,
                d.date_received,
                d.date_processed,
                d.status,
                COUNT(li.item_id) AS item_count,
                SUM(li.amount) AS total_amount
            FROM 
                documents d
            LEFT JOIN
                line_items li ON d.doc_id = li.doc_id
            WHERE
                d.doc_id = :doc_id
            GROUP BY
                d.doc_id, d.file_name, d.doc_type, d.party, d.date_created, d.date_received, d.date_processed, d.status
        """)
        
        result = self.db_session.execute(query, {"doc_id": doc_id}).fetchone()
        
        if not result:
            return None
            
        doc_id, file_name, doc_type, party, date_created, date_received, date_processed, status, item_count, total_amount = result
        
        # Handle different date formats - some might be datetime objects, others might be strings
        def format_date(date_val):
            if date_val is None:
                return None
            if hasattr(date_val, 'isoformat'):
                return date_val.isoformat()
            return str(date_val)
        
        return {
            'doc_id': doc_id,
            'file_name': file_name,
            'doc_type': doc_type,
            'party': party,
            'date': format_date(date_created),
            'date_received': format_date(date_received),
            'date_processed': format_date(date_processed),
            'status': status,
            'item_count': item_count,
            'total_amount': float(total_amount) if total_amount is not None else None
        }
    
    def _get_existing_relationships(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get existing relationships for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of existing relationships
        """
        query = text("""
            SELECT 
                r.relationship_id,
                r.source_doc_id,
                r.target_doc_id,
                r.relationship_type,
                r.confidence,
                d.doc_type,
                d.party,
                d.date_created
            FROM 
                document_relationships r
            JOIN 
                documents d ON (
                    CASE 
                        WHEN r.source_doc_id = :doc_id THEN r.target_doc_id = d.doc_id
                        ELSE r.source_doc_id = d.doc_id
                    END
                )
            WHERE 
                r.source_doc_id = :doc_id OR r.target_doc_id = :doc_id
        """)
        
        results = self.db_session.execute(query, {"doc_id": doc_id}).fetchall()
        
        relationships = []
        for result in results:
            relationship_id, source_doc_id, target_doc_id, relationship_type, confidence, doc_type, party, date_created = result
            
            # Determine which ID is the related document
            related_id = target_doc_id if source_doc_id == doc_id else source_doc_id
            
            # Determine relationship direction
            direction = "outgoing" if source_doc_id == doc_id else "incoming"
            
            relationships.append({
                'relationship_id': relationship_id,
                'source_doc_id': source_doc_id,
                'target_doc_id': target_doc_id,
                'related_doc_id': related_id,
                'relationship_type': relationship_type,
                'relationship_direction': direction,
                'confidence': float(confidence) if confidence is not None else None,
                'doc_type': doc_type,
                'party': party,
                'date': date_created.isoformat() if date_created else None,
                'is_existing': True
            })
        
        return relationships
    
    def _detect_potential_relationships(self, doc_id: str, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect potential relationships for a document.
        
        Args:
            doc_id: Document ID
            document: Document details
            
        Returns:
            List of potential relationships
        """
        # Get all other documents
        query = text("""
            SELECT 
                d.doc_id,
                d.doc_type,
                d.party,
                d.date_created,
                d.status,
                COUNT(li.item_id) AS item_count,
                SUM(li.amount) AS total_amount
            FROM 
                documents d
            LEFT JOIN
                line_items li ON d.doc_id = li.doc_id
            WHERE
                d.doc_id != :doc_id
            GROUP BY
                d.doc_id, d.doc_type, d.party, d.date_created, d.status
        """)
        
        other_docs = self.db_session.execute(query, {"doc_id": doc_id}).fetchall()
        
        # Convert to dictionaries for easier handling
        other_doc_dicts = []
        for result in other_docs:
            other_id, other_type, other_party, other_date, other_status, item_count, total_amount = result
            
            other_doc_dicts.append({
                'doc_id': other_id,
                'doc_type': other_type,
                'party': other_party,
                'date': other_date,
                'status': other_status,
                'item_count': item_count,
                'total_amount': float(total_amount) if total_amount is not None else None
            })
        
        # Detect relationships for each document
        potential_relationships = []
        
        for other_doc in other_doc_dicts:
            # Detect relationships between these two documents
            relationships = self._detect_relationship_between_documents(
                doc_id, document, 
                other_doc['doc_id'], other_doc
            )
            
            potential_relationships.extend(relationships)
        
        return potential_relationships
    
    def _detect_relationship_between_documents(self, doc_id1: str, doc1: Dict[str, Any], 
                                           doc_id2: str, doc2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect relationships between two documents.
        
        Args:
            doc_id1: First document ID
            doc1: First document details
            doc_id2: Second document ID
            doc2: Second document details
            
        Returns:
            List of detected relationships
        """
        relationships = []
        
        # Skip if missing critical data
        if not doc1 or not doc2:
            return relationships
            
        # Get document dates
        date1 = None
        if doc1.get('date'):
            try:
                date1 = datetime.fromisoformat(doc1['date'].replace('Z', '+00:00'))
            except (ValueError, TypeError):
                if isinstance(doc1['date'], datetime):
                    date1 = doc1['date']
        
        date2 = None
        if doc2.get('date'):
            try:
                date2 = datetime.fromisoformat(doc2['date'].replace('Z', '+00:00'))
            except (ValueError, TypeError):
                if isinstance(doc2['date'], datetime):
                    date2 = doc2['date']
        
        # Check document types
        type1 = doc1.get('doc_type', '').lower() if doc1.get('doc_type') else ''
        type2 = doc2.get('doc_type', '').lower() if doc2.get('doc_type') else ''
        
        # Get document status for rejection detection
        status1 = doc1.get('status', '').lower() if doc1.get('status') else ''
        status2 = doc2.get('status', '').lower() if doc2.get('status') else ''
        
        # Check for specific relationships based on document types
        if type1 == 'change_order' and type2 == 'correspondence':
            # Change order and correspondence - check specifically for rejection relationship
            if 'reject' in status2 or 'denied' in status2 or 'decline' in status2:
                relationships.append({
                    'source_doc_id': doc_id1,
                    'target_doc_id': doc_id2,
                    'relationship_type': 'rejection',
                    'confidence': 0.95,
                    'explanation': 'Change order rejected in correspondence'
                })
            else:
                relationships.append({
                    'source_doc_id': doc_id1,
                    'target_doc_id': doc_id2,
                    'relationship_type': 'correspondence',
                    'confidence': 0.85,
                    'explanation': 'Change order and related correspondence'
                })
        elif type2 == 'change_order' and type1 == 'correspondence':
            # Correspondence and change order - check specifically for rejection relationship
            if 'reject' in status1 or 'denied' in status1 or 'decline' in status1:
                relationships.append({
                    'source_doc_id': doc_id2,
                    'target_doc_id': doc_id1,
                    'relationship_type': 'rejection',
                    'confidence': 0.95,
                    'explanation': 'Change order rejected in correspondence'
                })
            else:
                relationships.append({
                    'source_doc_id': doc_id2,
                    'target_doc_id': doc_id1,
                    'relationship_type': 'correspondence',
                    'confidence': 0.85,
                    'explanation': 'Change order and related correspondence'
                })
        
        # Check for relationships between payment applications and change orders
        if (type1 == 'payment_app' and type2 == 'change_order') or \
           (type2 == 'payment_app' and type1 == 'change_order'):
            # Determine which is the payment app and which is the change order
            if type1 == 'payment_app':
                pay_app_id, co_id = doc_id1, doc_id2
                pay_app = doc1
                co = doc2
                co_status = status2
            else:
                pay_app_id, co_id = doc_id2, doc_id1
                pay_app = doc2
                co = doc1
                co_status = status1
                
            # Standard relationship
            relationships.append({
                'source_doc_id': co_id,
                'target_doc_id': pay_app_id,
                'relationship_type': 'referenced_in',
                'confidence': 0.8,
                'explanation': 'Change order referenced in payment application'
            })
            
            # Check if this might be a rejected change order appearing in payment app
            if 'reject' in co_status or 'denied' in co_status or 'decline' in co_status:
                relationships.append({
                    'source_doc_id': co_id,
                    'target_doc_id': pay_app_id,
                    'relationship_type': 'rejected_reappearing',
                    'confidence': 0.9,
                    'explanation': 'Rejected change order items appear in payment application'
                })
        
        # Check for relationships between invoices and change orders/payment apps
        if 'invoice' in type1 or 'invoice' in type2:
            # Determine which is the invoice
            if 'invoice' in type1:
                invoice_id, other_id = doc_id1, doc_id2
                invoice = doc1
                other = doc2
                other_type = type2
                other_status = status2
            else:
                invoice_id, other_id = doc_id2, doc_id1
                invoice = doc2
                other = doc1
                other_type = type1
                other_status = status1
                
            rel_type = 'related_invoice'
            explanation = f'Invoice related to {other_type}'
            
            relationships.append({
                'source_doc_id': other_id,
                'target_doc_id': invoice_id,
                'relationship_type': rel_type,
                'confidence': 0.85,
                'explanation': explanation
            })
            
            # Check if this might be a rejected change order appearing in invoice
            if other_type == 'change_order' and ('reject' in other_status or 'denied' in other_status or 'decline' in other_status):
                relationships.append({
                    'source_doc_id': other_id,
                    'target_doc_id': invoice_id,
                    'relationship_type': 'rejected_reappearing',
                    'confidence': 0.9,
                    'explanation': 'Rejected change order items appear in invoice'
                })
            
            # Also check for invoice being referenced in payment application
            if other_type == 'payment_app':
                relationships.append({
                    'source_doc_id': invoice_id,
                    'target_doc_id': other_id,
                    'relationship_type': 'referenced_in',
                    'confidence': 0.85,
                    'explanation': 'Invoice referenced in payment application'
                })
        
        # Check for parent-child relationship
        if date1 and date2:
            # If documents are sequential in time (within a reasonable time frame)
            time_diff = abs((date2 - date1).days)
            
            if time_diff <= 30:  # Within a month
                # Check for specific type patterns
                
                # Change order -> approval
                if (type1 == 'change_order' and 
                    ('approval' in type2 or 'accept' in type2 or 'correspondence' in type2) and 
                    date1 < date2):
                    
                    relationships.append({
                        'source_doc_id': doc_id1,
                        'target_doc_id': doc_id2,
                        'relationship_type': 'approval',
                        'confidence': 0.9,
                        'explanation': 'Change order followed by approval document'
                    })
                
                # Change order -> rejection
                elif (type1 == 'change_order' and 
                      ('reject' in type2 or 'denial' in type2 or 'correspondence' in type2) and 
                      date1 < date2):
                    
                    relationships.append({
                        'source_doc_id': doc_id1,
                        'target_doc_id': doc_id2,
                        'relationship_type': 'rejection',
                        'confidence': 0.9,
                        'explanation': 'Change order followed by rejection document'
                    })
                
                # Document -> revision
                elif (type1 == type2 and date1 < date2):
                    file_name1 = doc1.get('file_name', '').lower()
                    file_name2 = doc2.get('file_name', '').lower()
                    
                    # Check if file names suggest a revision
                    is_revision = False
                    common_strings = ['rev', 'version', 'v1', 'v2', 'v3', 'update']
                    
                    for common in common_strings:
                        if common in file_name2 and common not in file_name1:
                            is_revision = True
                            break
                    
                    if is_revision:
                        relationships.append({
                            'source_doc_id': doc_id1,
                            'target_doc_id': doc_id2,
                            'relationship_type': 'revision',
                            'confidence': 0.85,
                            'explanation': 'Earlier document and a likely revision'
                        })
                
                # Sequential payment applications
                elif type1 == 'payment_app' and type2 == 'payment_app' and date1 < date2:
                    # Extract payment application numbers if possible
                    file_name1 = doc1.get('file_name', '')
                    file_name2 = doc2.get('file_name', '')
                    
                    # Try to extract sequential numbering
                    is_sequential = False
                    if '_PayApp_' in file_name1 and '_PayApp_' in file_name2:
                        try:
                            num1 = int(file_name1.split('_PayApp_')[1].split('.')[0])
                            num2 = int(file_name2.split('_PayApp_')[1].split('.')[0])
                            if num2 == num1 + 1:
                                is_sequential = True
                        except (ValueError, IndexError):
                            pass
                    
                    relationships.append({
                        'source_doc_id': doc_id1,
                        'target_doc_id': doc_id2,
                        'relationship_type': 'sequential',
                        'confidence': 0.95 if is_sequential else 0.8,
                        'explanation': 'Sequential payment applications' if is_sequential else 'Related payment applications'
                    })
                
                # Sequential change orders
                elif type1 == 'change_order' and type2 == 'change_order' and date1 < date2:
                    # Extract CO numbers if possible
                    file_name1 = doc1.get('file_name', '')
                    file_name2 = doc2.get('file_name', '')
                    
                    # Try to extract sequential numbering
                    is_sequential = False
                    if '_CO_' in file_name1 and '_CO_' in file_name2:
                        try:
                            num1 = int(file_name1.split('_CO_')[1].split('.')[0])
                            num2 = int(file_name2.split('_CO_')[1].split('.')[0])
                            if num2 == num1 + 1:
                                is_sequential = True
                        except (ValueError, IndexError):
                            pass
                    
                    relationships.append({
                        'source_doc_id': doc_id1,
                        'target_doc_id': doc_id2,
                        'relationship_type': 'sequential',
                        'confidence': 0.95 if is_sequential else 0.8,
                        'explanation': 'Sequential change orders' if is_sequential else 'Related change orders'
                    })
                
                # Parent -> child based on timing and type
                elif date1 < date2:
                    confidence = 0.75
                    relationships.append({
                        'source_doc_id': doc_id1,
                        'target_doc_id': doc_id2,
                        'relationship_type': 'parent_child',
                        'confidence': confidence,
                        'explanation': 'Earlier document potentially related to later document'
                    })
        
        # Check for amendment relationship
        if 'amendment' in type2 and date1 and date2 and date1 < date2:
            relationships.append({
                'source_doc_id': doc_id1,
                'target_doc_id': doc_id2,
                'relationship_type': 'amendment',
                'confidence': 0.9,
                'explanation': 'Original document and an amendment'
            })
        
        # Check for response relationship
        if (('letter' in type1 or 'request' in type1 or 'correspondence' in type1) and 
            ('response' in type2 or 'reply' in type2 or 'correspondence' in type2) and 
            date1 and date2 and date1 < date2):
            
            relationships.append({
                'source_doc_id': doc_id1,
                'target_doc_id': doc_id2,
                'relationship_type': 'response',
                'confidence': 0.9,
                'explanation': 'Request document and a response'
            })
        
        return relationships
    
    def detect_and_register_all_relationships(self) -> Dict[str, Any]:
        """Detect and register relationships between all documents in the database.
        
        This function analyzes relationships between all documents and registers
        them in the database to improve network analysis.
        
        Returns:
            Dictionary with relationship detection results
        """
        logger.info("Detecting and registering relationships between all documents")
        
        # Get all documents
        query = text("""
            SELECT 
                d.doc_id,
                d.doc_type,
                d.party,
                d.date_created,
                d.status,
                d.file_name
            FROM 
                documents d
            ORDER BY
                d.date_created
        """)
        
        results = self.db_session.execute(query).fetchall()
        
        # Convert to dictionary for easier processing
        documents = []
        for result in results:
            doc_id, doc_type, party, date_created, status, file_name = result
            
            documents.append({
                'doc_id': doc_id,
                'doc_type': doc_type,
                'party': party,
                'date': date_created,
                'status': status,
                'file_name': file_name
            })
        
        if not documents:
            logger.warning("No documents found in the database")
            return {'error': 'No documents found', 'created_count': 0}
        
        # Check if the document relationships table is empty
        count_query = text("SELECT COUNT(*) FROM document_relationships")
        existing_count = self.db_session.execute(count_query).scalar()
        
        # Process all possible document pairs to detect relationships
        created_count = 0
        detected_relationships = []
        
        # Also track change orders and rejected change orders for circular reference detection
        rejected_change_orders = []
        change_orders = []
        payment_apps = []
        
        # First identify all change orders, rejected ones, and payment applications
        for doc in documents:
            doc_type = doc.get('doc_type', '').lower()
            doc_status = doc.get('status', '').lower()
            
            if doc_type == 'change_order':
                change_orders.append(doc)
                if 'reject' in doc_status or 'denied' in doc_status or 'decline' in doc_status:
                    rejected_change_orders.append(doc)
            elif doc_type == 'payment_app':
                payment_apps.append(doc)
                
        # Process documents in chronological order
        for i, doc1 in enumerate(documents):
            for doc2 in documents[i+1:]:  # Only process each pair once
                # Detect relationships between these two documents
                relationships = self._detect_relationship_between_documents(
                    doc1['doc_id'], doc1, 
                    doc2['doc_id'], doc2
                )
                
                # Register relationships if they meet confidence threshold
                for rel in relationships:
                    if rel['confidence'] >= self.min_confidence:
                        # Register in the database
                        result = self._register_relationship(
                            rel['source_doc_id'], 
                            rel['target_doc_id'], 
                            rel['relationship_type'], 
                            rel['confidence']
                        )
                        
                        if result.get('status') == 'created':
                            created_count += 1
                            detected_relationships.append(rel)
        
        # Also check for shared item relationships
        for doc in documents:
            shared_items = self._find_shared_item_relationships(doc['doc_id'])
            
            for rel in shared_items:
                # Register in the database
                result = self._register_relationship(
                    rel['source_doc_id'], 
                    rel['target_doc_id'], 
                    rel['relationship_type'], 
                    rel['confidence']
                )
                
                if result.get('status') == 'created':
                    created_count += 1
                    detected_relationships.append(rel)
        
        # Detect splitting large rejected amounts into smaller change orders
        # Only when we have rejected change orders
        if rejected_change_orders and len(change_orders) > 3:  # Need enough change orders for a pattern
            for rejected_co in rejected_change_orders:
                # Get rejected amount
                rejected_amount = rejected_co.get('total_amount', 0)
                if rejected_amount and rejected_amount > 0:
                    # Find small change orders that came after the rejection
                    small_cos = []
                    for co in change_orders:
                        # Skip the rejected one itself
                        if co['doc_id'] == rejected_co['doc_id']:
                            continue
                            
                        # Check if this is a small change order that came after the rejected one
                        co_amount = co.get('total_amount', 0)
                        
                        # Check dates to ensure chronological order
                        rejected_date = self._parse_date(rejected_co.get('date'))
                        co_date = self._parse_date(co.get('date'))
                        
                        if (co_amount and 
                            co_amount > 0 and 
                            co_amount < rejected_amount * 0.4 and  # Significantly smaller
                            rejected_date and co_date and
                            co_date > rejected_date):  # Later date
                            
                            small_cos.append({
                                'doc_id': co['doc_id'],
                                'amount': co_amount,
                                'date': co_date
                            })
                    
                    # Check if small COs add up to near the rejected amount
                    if small_cos:
                        total_small_amount = sum(co['amount'] for co in small_cos)
                        percentage = (total_small_amount / rejected_amount) * 100
                        
                        # If the total is close to the rejected amount (80-120%)
                        if 80 <= percentage <= 120:
                            # Create circular reference relationships between the small change orders
                            # and from the rejected change order to each small one
                            for small_co in small_cos:
                                # Register relationship from rejected to small CO
                                rel = {
                                    'source_doc_id': rejected_co['doc_id'],
                                    'target_doc_id': small_co['doc_id'],
                                    'relationship_type': 'sequential_change_orders',
                                    'confidence': 0.9,
                                    'explanation': 'Large rejected change order split into smaller change orders'
                                }
                                
                                # Register in the database
                                result = self._register_relationship(
                                    rel['source_doc_id'], 
                                    rel['target_doc_id'], 
                                    rel['relationship_type'], 
                                    rel['confidence']
                                )
                                
                                if result.get('status') == 'created':
                                    created_count += 1
                                    detected_relationships.append(rel)
                            
                            # Connect the small change orders to each other to create a circular reference
                            if len(small_cos) > 1:
                                for i in range(len(small_cos) - 1):
                                    rel = {
                                        'source_doc_id': small_cos[i]['doc_id'],
                                        'target_doc_id': small_cos[i+1]['doc_id'],
                                        'relationship_type': 'circular_reference',
                                        'confidence': 0.95,
                                        'explanation': 'Sequential small change orders from split rejected change order'
                                    }
                                    
                                    # Register in the database
                                    result = self._register_relationship(
                                        rel['source_doc_id'], 
                                        rel['target_doc_id'], 
                                        rel['relationship_type'], 
                                        rel['confidence']
                                    )
                                    
                                    if result.get('status') == 'created':
                                        created_count += 1
                                        detected_relationships.append(rel)
                                
                                # Complete the circle
                                rel = {
                                    'source_doc_id': small_cos[-1]['doc_id'],
                                    'target_doc_id': small_cos[0]['doc_id'],
                                    'relationship_type': 'circular_reference',
                                    'confidence': 0.95,
                                    'explanation': 'Circular reference between split change orders'
                                }
                                
                                # Register in the database
                                result = self._register_relationship(
                                    rel['source_doc_id'], 
                                    rel['target_doc_id'], 
                                    rel['relationship_type'], 
                                    rel['confidence']
                                )
                                
                                if result.get('status') == 'created':
                                    created_count += 1
                                    detected_relationships.append(rel)
        
        logger.info(f"Detected and registered {created_count} new document relationships")
        
        return {
            'document_count': len(documents),
            'created_count': created_count,
            'existing_count': existing_count,
            'detected_relationships': detected_relationships
        }
    
    def _find_shared_item_relationships(self, doc_id: str) -> List[Dict[str, Any]]:
        """Find documents that share line items with a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of shared item relationships
        """
        # First, get the line items for this document
        query = text("""
            SELECT 
                li.amount,
                li.description
            FROM 
                line_items li
            WHERE 
                li.doc_id = :doc_id
                AND li.amount IS NOT NULL
        """)
        
        items = self.db_session.execute(query, {"doc_id": doc_id}).fetchall()
        
        # Skip if no items
        if not items:
            return []
            
        # Find documents with similar items
        shared_items = {}
        
        for amount, description in items:
            # Skip if invalid data
            if amount is None:
                continue
                
            # Find documents with the same amount
            match_query = text("""
                SELECT 
                    li.doc_id,
                    COUNT(*) AS match_count
                FROM 
                    line_items li
                WHERE 
                    li.doc_id != :doc_id
                    AND li.amount BETWEEN :amount_min AND :amount_max
                GROUP BY 
                    li.doc_id
            """)
            
            tolerance = amount * 0.01  # 1% tolerance
            amount_min = amount - tolerance
            amount_max = amount + tolerance
            
            matches = self.db_session.execute(
                match_query, 
                {
                    "doc_id": doc_id, 
                    "amount_min": amount_min, 
                    "amount_max": amount_max
                }
            ).fetchall()
            
            for match_doc_id, match_count in matches:
                if match_doc_id not in shared_items:
                    shared_items[match_doc_id] = {
                        'doc_id': match_doc_id,
                        'match_count': 0,
                        'matched_amounts': []
                    }
                
                shared_items[match_doc_id]['match_count'] += match_count
                shared_items[match_doc_id]['matched_amounts'].append(float(amount))
        
        # Get document details for matched documents
        relationships = []
        for doc_id2, match_data in shared_items.items():
            # Skip if too few matches
            if match_data['match_count'] < 2:
                continue
                
            # Get document details
            doc2 = self._get_document(doc_id2)
            if not doc2:
                continue
                
            # Calculate confidence based on match count
            confidence = min(0.95, 0.7 + min(match_data['match_count'] / 10, 0.25))
            
            if confidence >= self.min_confidence:
                relationships.append({
                    'source_doc_id': doc_id,
                    'target_doc_id': doc_id2,
                    'relationship_type': 'contains_same_items',
                    'match_count': match_data['match_count'],
                    'matched_amounts': match_data['matched_amounts'],
                    'confidence': confidence,
                    'doc_type': doc2.get('doc_type'),
                    'party': doc2.get('party'),
                    'date': doc2.get('date'),
                    'explanation': f"Documents share {match_data['match_count']} matching line items"
                })
        
        return relationships
    
    def _register_new_relationships(self, doc_id: str, potential_relationships: List[Dict[str, Any]], 
                               existing_relationships: List[Dict[str, Any]]) -> None:
        """Register new relationships in the database.
        
        Args:
            doc_id: Document ID
            potential_relationships: List of potential relationships
            existing_relationships: List of existing relationships
        """
        # Create a set of existing relationship keys
        existing_keys = set()
        for rel in existing_relationships:
            key = (rel['source_doc_id'], rel['target_doc_id'], rel['relationship_type'])
            existing_keys.add(key)
        
        # Register new relationships
        for rel in potential_relationships:
            key = (rel['source_doc_id'], rel['target_doc_id'], rel['relationship_type'])
            
            # Skip if already exists
            if key in existing_keys:
                continue
                
            # Register relationship
            if rel['confidence'] >= self.min_confidence:
                self._register_relationship(
                    rel['source_doc_id'], 
                    rel['target_doc_id'], 
                    rel['relationship_type'], 
                    rel['confidence']
                )
    
    def _parse_date(self, date_value):
        """Parse date value into a datetime object.
        
        Args:
            date_value: Date value (string, datetime, or None)
            
        Returns:
            datetime object or None
        """
        if date_value is None:
            return None
            
        # If it's already a datetime object, return it
        from datetime import datetime
        if isinstance(date_value, datetime):
            return date_value
            
        # If it's a string, try to parse it
        if isinstance(date_value, str):
            try:
                # Try different date formats
                formats = [
                    '%Y-%m-%d',       # 2023-03-15
                    '%Y-%m-%dT%H:%M:%S',  # ISO format
                    '%Y-%m-%dT%H:%M:%S.%f',  # ISO format with microseconds
                    '%B %d, %Y',      # March 15, 2023
                    '%b %d, %Y',      # Mar 15, 2023
                    '%m/%d/%Y',       # 03/15/2023
                    '%d-%m-%Y',       # 15-03-2023
                ]
                
                for fmt in formats:
                    try:
                        return datetime.strptime(date_value, fmt)
                    except ValueError:
                        continue
            except Exception:
                pass
                
        return None
            
    def _register_relationship(self, source_doc_id: str, target_doc_id: str, 
                          relationship_type: str, confidence: float = 1.0) -> Dict[str, Any]:
        """Register a relationship between two documents in the database.
        
        Args:
            source_doc_id: Source document ID
            target_doc_id: Target document ID
            relationship_type: Type of relationship
            confidence: Confidence score for the relationship
            
        Returns:
            Dictionary with relationship information
        """
        logger.info(f"Registering relationship from {source_doc_id} to {target_doc_id}: {relationship_type}")
        
        # Check if relationship already exists
        check_query = text("""
            SELECT 
                relationship_id
            FROM 
                document_relationships
            WHERE 
                source_doc_id = :source_id
                AND target_doc_id = :target_id
                AND relationship_type = :rel_type
        """)
        
        result = self.db_session.execute(
            check_query, 
            {
                "source_id": source_doc_id, 
                "target_id": target_doc_id, 
                "rel_type": relationship_type
            }
        ).fetchone()
        
        if result:
            # Update existing relationship
            relationship_id = result[0]
            
            update_query = text("""
                UPDATE document_relationships 
                SET 
                    confidence = :confidence
                WHERE 
                    relationship_id = :rel_id
                RETURNING relationship_id
            """)
            
            try:
                self.db_session.execute(
                    update_query, 
                    {
                        "rel_id": relationship_id, 
                        "confidence": confidence
                    }
                )
                
                self.db_session.commit()
                
                return {
                    'relationship_id': relationship_id,
                    'source_doc_id': source_doc_id,
                    'target_doc_id': target_doc_id,
                    'relationship_type': relationship_type,
                    'confidence': confidence,
                    'status': 'updated'
                }
            except Exception as e:
                self.db_session.rollback()
                logger.exception(f"Error updating relationship: {e}")
                return {'error': str(e)}
        else:
            # Create new relationship
            insert_query = text("""
                INSERT INTO document_relationships 
                    (source_doc_id, target_doc_id, relationship_type, confidence) 
                VALUES 
                    (:source_id, :target_id, :rel_type, :confidence)
                RETURNING relationship_id
            """)
            
            try:
                result = self.db_session.execute(
                    insert_query, 
                    {
                        "source_id": source_doc_id, 
                        "target_id": target_doc_id, 
                        "rel_type": relationship_type, 
                        "confidence": confidence
                    }
                ).fetchone()
                
                self.db_session.commit()
                
                return {
                    'relationship_id': result[0],
                    'source_doc_id': source_doc_id,
                    'target_doc_id': target_doc_id,
                    'relationship_type': relationship_type,
                    'confidence': confidence,
                    'status': 'created'
                }
            except Exception as e:
                self.db_session.rollback()
                logger.exception(f"Error creating relationship: {e}")
                return {'error': str(e)}
    
    def _build_relationship_graph(self, documents: Dict[str, Dict[str, Any]], 
                             relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a relationship graph.
        
        Args:
            documents: Dictionary of document details
            relationships: List of relationships
            
        Returns:
            Relationship graph
        """
        # Create nodes
        nodes = []
        for doc_id, doc in documents.items():
            nodes.append({
                'id': doc_id,
                'doc_type': doc.get('doc_type'),
                'party': doc.get('party'),
                'date': doc.get('date'),
                'total_amount': doc.get('total_amount')
            })
        
        # Create edges
        edges = []
        for rel in relationships:
            # Skip if either node is missing
            if rel['source_doc_id'] not in documents or rel['target_doc_id'] not in documents:
                continue
                
            edges.append({
                'source': rel['source_doc_id'],
                'target': rel['target_doc_id'],
                'type': rel['relationship_type'],
                'confidence': rel.get('confidence', 1.0)
            })
        
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    def _find_previous_document(self, doc_id: str, visited: Set[str]) -> Optional[str]:
        """Find the previous document in a chain.
        
        Args:
            doc_id: Document ID
            visited: Set of already visited document IDs
            
        Returns:
            Previous document ID or None
        """
        # Look for parent-child, revision, or amendment relationships where
        # this document is the target (child)
        query = text("""
            SELECT 
                r.source_doc_id,
                r.relationship_type,
                r.confidence,
                d.date_created
            FROM 
                document_relationships r
            JOIN 
                documents d ON r.source_doc_id = d.doc_id
            WHERE 
                r.target_doc_id = :doc_id
                AND r.relationship_type IN ('parent_child', 'revision', 'amendment')
                AND r.confidence >= :min_confidence
            ORDER BY 
                r.confidence DESC, d.date_created DESC
        """)
        
        results = self.db_session.execute(query, {"doc_id": doc_id, "min_confidence": self.min_confidence}).fetchall()
        
        # Filter out already visited documents
        candidates = []
        for result in results:
            source_id, rel_type, confidence, date_created = result
            
            if source_id not in visited:
                candidates.append({
                    'doc_id': source_id,
                    'relationship_type': rel_type,
                    'confidence': confidence,
                    'date': date_created
                })
        
        # Return the best candidate (highest confidence)
        if candidates:
            return candidates[0]['doc_id']
            
        return None
    
    def _find_next_document(self, doc_id: str, visited: Set[str]) -> Optional[str]:
        """Find the next document in a chain.
        
        Args:
            doc_id: Document ID
            visited: Set of already visited document IDs
            
        Returns:
            Next document ID or None
        """
        # Look for parent-child, revision, or amendment relationships where
        # this document is the source (parent)
        query = text("""
            SELECT 
                r.target_doc_id,
                r.relationship_type,
                r.confidence,
                d.date_created
            FROM 
                document_relationships r
            JOIN 
                documents d ON r.target_doc_id = d.doc_id
            WHERE 
                r.source_doc_id = :doc_id
                AND r.relationship_type IN ('parent_child', 'revision', 'amendment')
                AND r.confidence >= :min_confidence
            ORDER BY 
                r.confidence DESC, d.date_created ASC
        """)
        
        results = self.db_session.execute(query, {"doc_id": doc_id, "min_confidence": self.min_confidence}).fetchall()
        
        # Filter out already visited documents
        candidates = []
        for result in results:
            target_id, rel_type, confidence, date_created = result
            
            if target_id not in visited:
                candidates.append({
                    'doc_id': target_id,
                    'relationship_type': rel_type,
                    'confidence': confidence,
                    'date': date_created
                })
        
        # Return the best candidate (highest confidence and earliest date)
        if candidates:
            return candidates[0]['doc_id']
            
        return None
    
    def _find_related_documents(self, doc_id: str, visited: Set[str]) -> List[str]:
        """Find related documents that are not in the main chain.
        
        Args:
            doc_id: Document ID
            visited: Set of already visited document IDs
            
        Returns:
            List of related document IDs
        """
        # Look for relationships other than parent-child, revision, amendment
        query = text("""
            SELECT 
                CASE 
                    WHEN r.source_doc_id = :doc_id THEN r.target_doc_id
                    ELSE r.source_doc_id
                END AS related_id,
                r.relationship_type,
                r.confidence
            FROM 
                document_relationships r
            WHERE 
                (r.source_doc_id = :doc_id OR r.target_doc_id = :doc_id)
                AND r.relationship_type NOT IN ('parent_child', 'revision', 'amendment')
                AND r.confidence >= :min_confidence
            ORDER BY 
                r.confidence DESC
        """)
        
        results = self.db_session.execute(query, {"doc_id": doc_id, "min_confidence": self.min_confidence}).fetchall()
        
        # Filter out already visited documents
        related_ids = []
        for result in results:
            related_id, rel_type, confidence = result
            
            if related_id not in visited:
                related_ids.append(related_id)
        
        return related_ids