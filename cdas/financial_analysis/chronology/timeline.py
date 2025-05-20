"""
Timeline analyzer for the financial analysis engine.

This module provides timeline analysis capabilities, tracking financial events
over time and identifying key points in financial chronologies.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
from decimal import Decimal

import logging
logger = logging.getLogger(__name__)


class TimelineAnalyzer:
    """Analyzes financial timelines and chronologies."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the timeline analyzer.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Default configuration
        self.min_confidence = self.config.get('min_confidence', 0.7)
        
        # Key event definitions
        self.key_event_types = {
            'first_occurrence': "First appearance of an amount",
            'status_change': "Change in item status",
            'document_type_change': "Movement between document types",
            'party_change': "Change between parties",
            'significant_gap': "Unusually long gap between events",
            'final_occurrence': "Final appearance of an amount"
        }
        
        # Timeline configuration
        self.significant_gap_days = self.config.get('significant_gap_days', 30)
        
    def analyze_history(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the history of a financial item across documents.
        
        Args:
            matches: List of matches for the item
            
        Returns:
            Chronological analysis
        """
        logger.info(f"Analyzing history for {len(matches)} matches")
        
        # Sort matches by date
        sorted_matches = self._sort_by_date(matches)
        
        # Create timeline of events
        timeline = self._create_timeline(sorted_matches)
        
        # Identify key events in the timeline
        key_events = self._identify_key_events(timeline)
        
        # Calculate financial journey
        financial_journey = self._calculate_financial_journey(timeline)
        
        # Generate timeline summary
        summary = self._generate_timeline_summary(timeline, key_events, financial_journey)
        
        # Return analysis results
        result = {
            'timeline': timeline,
            'key_events': key_events,
            'financial_journey': financial_journey,
            'summary': summary
        }
        
        logger.info(f"Timeline analysis complete: {len(timeline)} events, {len(key_events)} key events")
        return result
    
    def analyze_document_timeline(self, doc_id: str) -> Dict[str, Any]:
        """Analyze the timeline of a document and related documents.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Timeline analysis
        """
        logger.info(f"Analyzing document timeline for document: {doc_id}")
        
        # Get document details
        doc = self._get_document(doc_id)
        if not doc:
            logger.warning(f"Document not found: {doc_id}")
            return {'error': 'Document not found'}
        
        # Get related documents
        related_docs = self._get_related_documents(doc_id)
        
        # Build document timeline
        doc_timeline = self._create_document_timeline(doc, related_docs)
        
        # Identify key documents in the timeline
        key_documents = self._identify_key_documents(doc_timeline)
        
        # Generate document sequence
        document_sequence = self._generate_document_sequence(doc_timeline)
        
        # Return analysis results
        result = {
            'document': doc,
            'document_timeline': doc_timeline,
            'key_documents': key_documents,
            'document_sequence': document_sequence
        }
        
        logger.info(f"Document timeline analysis complete: {len(doc_timeline)} documents, {len(key_documents)} key documents")
        return result
    
    def analyze_multi_document_timeline(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Analyze the timeline across multiple specified documents.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            Timeline analysis
        """
        logger.info(f"Analyzing timeline across {len(doc_ids)} documents")
        
        # Get details for all documents
        docs = []
        for doc_id in doc_ids:
            doc = self._get_document(doc_id)
            if doc:
                docs.append(doc)
        
        if not docs:
            logger.warning("No valid documents found")
            return {'error': 'No valid documents found'}
        
        # Build document timeline
        doc_timeline = self._create_multi_document_timeline(docs)
        
        # Identify key documents in the timeline
        key_documents = self._identify_key_documents(doc_timeline)
        
        # Generate document sequence
        document_sequence = self._generate_document_sequence(doc_timeline)
        
        # Return analysis results
        result = {
            'documents': docs,
            'document_timeline': doc_timeline,
            'key_documents': key_documents,
            'document_sequence': document_sequence
        }
        
        logger.info(f"Multi-document timeline analysis complete")
        return result
    
    def get_project_timeline(self) -> Dict[str, Any]:
        """Get a comprehensive timeline of all documents in the project.
        
        Returns:
            Project timeline
        """
        logger.info("Generating project timeline")
        
        # Get all documents with dates
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
                d.date_created IS NOT NULL
            GROUP BY
                d.doc_id, d.file_name, d.doc_type, d.party, d.date_created, d.date_received, d.date_processed, d.status
            ORDER BY
                d.date_created
        """)
        
        docs = self.db_session.execute(query).fetchall()
        
        # Convert to dictionaries and create timeline
        timeline = []
        for doc in docs:
            doc_id, file_name, doc_type, party, date_created, date_received, date_processed, status, item_count, total_amount = doc
            
            timeline.append({
                'event_type': 'document',
                'doc_id': doc_id,
                'file_name': file_name,
                'doc_type': doc_type,
                'party': party,
                'date': date_created.isoformat() if date_created else None,
                'date_received': date_received.isoformat() if date_received else None,
                'date_processed': date_processed.isoformat() if date_processed else None,
                'status': status,
                'item_count': item_count,
                'total_amount': float(total_amount) if total_amount is not None else None
            })
        
        # Identify key periods in the timeline
        key_periods = self._identify_key_periods(timeline)
        
        # Generate timeline summary
        summary = self._generate_project_timeline_summary(timeline, key_periods)
        
        # Return timeline information
        result = {
            'timeline': timeline,
            'key_periods': key_periods,
            'summary': summary,
            'document_count': len(timeline)
        }
        
        logger.info(f"Project timeline generated with {len(timeline)} documents")
        return result
    
    def _sort_by_date(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort matches by date.
        
        Args:
            matches: List of matches
            
        Returns:
            Sorted list of matches
        """
        # Convert ISO date strings to datetime objects for sorting
        matches_with_datetime = []
        for match in matches:
            match_copy = match.copy()
            
            # Handle date field which could be an ISO string
            date_str = match.get('date')
            if isinstance(date_str, str):
                try:
                    match_copy['date'] = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    match_copy['date'] = None
            
            matches_with_datetime.append(match_copy)
        
        # Sort by date (using datetime.min for None values)
        sorted_matches = sorted(
            matches_with_datetime, 
            key=lambda x: x.get('date') or datetime.min
        )
        
        # Convert back to ISO strings for the result
        for match in sorted_matches:
            if isinstance(match.get('date'), datetime):
                match['date'] = match['date'].isoformat()
        
        return sorted_matches
    
    def _create_timeline(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create a timeline from matches.
        
        Args:
            matches: List of matches
            
        Returns:
            Timeline of events
        """
        timeline = []
        
        for i, match in enumerate(matches):
            # Parse date from ISO format if needed
            date = match.get('date')
            if isinstance(date, str):
                try:
                    date = datetime.fromisoformat(date.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    date = None
            
            # Create event
            event = {
                'event_id': i,
                'event_type': 'amount_occurrence',
                'date': date.isoformat() if date else None,
                'item_id': match.get('item_id'),
                'doc_id': match.get('doc_id'),
                'doc_type': match.get('doc_type'),
                'party': match.get('party'),
                'amount': float(match.get('amount')) if match.get('amount') is not None else None,
                'description': match.get('description')
            }
            
            timeline.append(event)
        
        return timeline
    
    def _identify_key_events(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key events in a timeline.
        
        Args:
            timeline: Timeline of events
            
        Returns:
            List of key events
        """
        key_events = []
        
        # Skip if timeline is empty
        if not timeline:
            return key_events
            
        # Identify first occurrence
        key_events.append({
            'event_id': timeline[0]['event_id'],
            'type': 'first_occurrence',
            'date': timeline[0]['date'],
            'doc_id': timeline[0]['doc_id'],
            'doc_type': timeline[0]['doc_type'],
            'party': timeline[0]['party'],
            'explanation': self.key_event_types['first_occurrence']
        })
        
        # Identify last occurrence
        key_events.append({
            'event_id': timeline[-1]['event_id'],
            'type': 'final_occurrence',
            'date': timeline[-1]['date'],
            'doc_id': timeline[-1]['doc_id'],
            'doc_type': timeline[-1]['doc_type'],
            'party': timeline[-1]['party'],
            'explanation': self.key_event_types['final_occurrence']
        })
        
        # Look for status changes, document type changes, and party changes
        for i in range(1, len(timeline)):
            prev_event = timeline[i-1]
            curr_event = timeline[i]
            
            # Check for significant gap
            prev_date = None
            curr_date = None
            
            if prev_event['date']:
                try:
                    prev_date = datetime.fromisoformat(prev_event['date'].replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    prev_date = None
                    
            if curr_event['date']:
                try:
                    curr_date = datetime.fromisoformat(curr_event['date'].replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    curr_date = None
            
            if prev_date and curr_date:
                gap_days = (curr_date - prev_date).days
                if gap_days > self.significant_gap_days:
                    key_events.append({
                        'event_id': curr_event['event_id'],
                        'type': 'significant_gap',
                        'date': curr_event['date'],
                        'doc_id': curr_event['doc_id'],
                        'previous_date': prev_event['date'],
                        'gap_days': gap_days,
                        'explanation': f"{self.key_event_types['significant_gap']} ({gap_days} days)"
                    })
            
            # Check for document type change
            if prev_event['doc_type'] != curr_event['doc_type']:
                key_events.append({
                    'event_id': curr_event['event_id'],
                    'type': 'document_type_change',
                    'date': curr_event['date'],
                    'doc_id': curr_event['doc_id'],
                    'from_doc_type': prev_event['doc_type'],
                    'to_doc_type': curr_event['doc_type'],
                    'explanation': f"{self.key_event_types['document_type_change']} ({prev_event['doc_type']} to {curr_event['doc_type']})"
                })
            
            # Check for party change
            if prev_event['party'] != curr_event['party']:
                key_events.append({
                    'event_id': curr_event['event_id'],
                    'type': 'party_change',
                    'date': curr_event['date'],
                    'doc_id': curr_event['doc_id'],
                    'from_party': prev_event['party'],
                    'to_party': curr_event['party'],
                    'explanation': f"{self.key_event_types['party_change']} ({prev_event['party']} to {curr_event['party']})"
                })
        
        return key_events
    
    def _calculate_financial_journey(self, timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate financial metrics across the timeline.
        
        Args:
            timeline: Timeline of events
            
        Returns:
            Financial journey metrics
        """
        # Initialize metrics
        journey = {
            'first_date': None,
            'last_date': None,
            'duration_days': None,
            'document_types': set(),
            'parties': set(),
            'doc_count': len(timeline)
        }
        
        # Skip if timeline is empty
        if not timeline:
            return journey
            
        # Get dates from first and last events
        first_date = None
        last_date = None
        
        if timeline[0]['date']:
            try:
                first_date = datetime.fromisoformat(timeline[0]['date'].replace('Z', '+00:00'))
                journey['first_date'] = timeline[0]['date']
            except (ValueError, TypeError):
                first_date = None
                
        if timeline[-1]['date']:
            try:
                last_date = datetime.fromisoformat(timeline[-1]['date'].replace('Z', '+00:00'))
                journey['last_date'] = timeline[-1]['date']
            except (ValueError, TypeError):
                last_date = None
        
        # Calculate duration
        if first_date and last_date:
            journey['duration_days'] = (last_date - first_date).days
        
        # Collect document types and parties
        for event in timeline:
            if event['doc_type']:
                journey['document_types'].add(event['doc_type'])
            if event['party']:
                journey['parties'].add(event['party'])
        
        # Convert sets to lists for JSON serialization
        journey['document_types'] = list(journey['document_types'])
        journey['parties'] = list(journey['parties'])
        
        return journey
    
    def _generate_timeline_summary(self, timeline: List[Dict[str, Any]], 
                             key_events: List[Dict[str, Any]], 
                             financial_journey: Dict[str, Any]) -> str:
        """Generate a summary of the timeline.
        
        Args:
            timeline: Timeline of events
            key_events: List of key events
            financial_journey: Financial journey metrics
            
        Returns:
            Timeline summary
        """
        if not timeline:
            return "No timeline events available."
            
        # Format date range
        date_range = ""
        if financial_journey.get('first_date') and financial_journey.get('last_date'):
            first_date = financial_journey['first_date']
            last_date = financial_journey['last_date']
            
            if isinstance(first_date, str):
                first_date = first_date.split('T')[0]  # Just get the date part
            if isinstance(last_date, str):
                last_date = last_date.split('T')[0]  # Just get the date part
                
            date_range = f"from {first_date} to {last_date}"
            
            if financial_journey.get('duration_days'):
                date_range += f" ({financial_journey['duration_days']} days)"
        
        # Format document types
        doc_types = ', '.join(financial_journey.get('document_types', []))
        
        # Format parties
        parties = ', '.join(financial_journey.get('parties', []))
        
        # Build summary
        summary = f"Amount appears in {len(timeline)} documents {date_range}"
        
        if doc_types:
            summary += f", across document types: {doc_types}"
            
        if parties:
            summary += f", involving parties: {parties}"
            
        # Add key event information
        if key_events:
            key_event_types = set(event['type'] for event in key_events if event['type'] != 'first_occurrence' and event['type'] != 'final_occurrence')
            
            if key_event_types:
                key_event_desc = ', '.join(key_event_types)
                summary += f". Key events include: {key_event_desc}"
        
        return summary
    
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
    
    def _get_related_documents(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get documents related to a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of related documents
        """
        # First check document relationships table
        rel_query = text("""
            SELECT 
                r.source_doc_id,
                r.target_doc_id,
                r.relationship_type
            FROM 
                document_relationships r
            WHERE 
                r.source_doc_id = :doc_id
                OR r.target_doc_id = :doc_id
        """)
        
        relationships = self.db_session.execute(rel_query, {"doc_id": doc_id}).fetchall()
        
        # Get related document IDs from relationships
        related_doc_ids = set()
        for rel in relationships:
            source_doc_id, target_doc_id, _ = rel
            
            if source_doc_id != doc_id:
                related_doc_ids.add(source_doc_id)
            if target_doc_id != doc_id:
                related_doc_ids.add(target_doc_id)
        
        # Also check for similar documents of the same type/party
        doc = self._get_document(doc_id)
        if doc:
            similar_query = text("""
                SELECT 
                    d.doc_id
                FROM 
                    documents d
                WHERE 
                    d.doc_id != :doc_id
                    AND (
                        (d.doc_type = :doc_type AND d.doc_type IS NOT NULL)
                        OR (d.party = :party AND d.party IS NOT NULL)
                    )
                LIMIT 10
            """)
            
            similar_results = self.db_session.execute(
                similar_query, 
                {
                    "doc_id": doc_id, 
                    "doc_type": doc.get('doc_type'), 
                    "party": doc.get('party')
                }
            ).fetchall()
            
            for result in similar_results:
                related_doc_ids.add(result[0])
        
        # Get details for all related documents
        related_docs = []
        for related_id in related_doc_ids:
            related_doc = self._get_document(related_id)
            if related_doc:
                related_docs.append(related_doc)
        
        return related_docs
    
    def _create_document_timeline(self, doc: Dict[str, Any], 
                              related_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create a timeline of a document and related documents.
        
        Args:
            doc: Main document
            related_docs: List of related documents
            
        Returns:
            Document timeline
        """
        # Combine main document and related documents
        all_docs = [doc] + related_docs
        
        # Create timeline events
        timeline = []
        for i, document in enumerate(all_docs):
            # Parse date from ISO format if needed
            date = document.get('date')
            if isinstance(date, str):
                try:
                    date = datetime.fromisoformat(date.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    date = None
            
            # Create event
            event = {
                'event_id': i,
                'event_type': 'document',
                'date': date.isoformat() if date else None,
                'doc_id': document.get('doc_id'),
                'file_name': document.get('file_name'),
                'doc_type': document.get('doc_type'),
                'party': document.get('party'),
                'status': document.get('status'),
                'item_count': document.get('item_count'),
                'total_amount': document.get('total_amount'),
                'is_main_document': document.get('doc_id') == doc.get('doc_id')
            }
            
            timeline.append(event)
        
        # Sort by date
        timeline = sorted(
            timeline, 
            key=lambda x: datetime.fromisoformat(x['date'].replace('Z', '+00:00')) if x['date'] else datetime.max
        )
        
        return timeline
    
    def _create_multi_document_timeline(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create a timeline from multiple documents.
        
        Args:
            docs: List of documents
            
        Returns:
            Document timeline
        """
        # Create timeline events
        timeline = []
        for i, document in enumerate(docs):
            # Parse date from ISO format if needed
            date = document.get('date')
            if isinstance(date, str):
                try:
                    date = datetime.fromisoformat(date.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    date = None
            
            # Create event
            event = {
                'event_id': i,
                'event_type': 'document',
                'date': date.isoformat() if date else None,
                'doc_id': document.get('doc_id'),
                'file_name': document.get('file_name'),
                'doc_type': document.get('doc_type'),
                'party': document.get('party'),
                'status': document.get('status'),
                'item_count': document.get('item_count'),
                'total_amount': document.get('total_amount')
            }
            
            timeline.append(event)
        
        # Sort by date
        timeline = sorted(
            timeline, 
            key=lambda x: datetime.fromisoformat(x['date'].replace('Z', '+00:00')) if x['date'] else datetime.max
        )
        
        return timeline
    
    def _identify_key_documents(self, doc_timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key documents in a document timeline.
        
        Args:
            doc_timeline: Document timeline
            
        Returns:
            List of key documents
        """
        key_documents = []
        
        # Skip if timeline is empty
        if not doc_timeline:
            return key_documents
            
        # Identify first document
        key_documents.append({
            'event_id': doc_timeline[0]['event_id'],
            'type': 'first_document',
            'date': doc_timeline[0]['date'],
            'doc_id': doc_timeline[0]['doc_id'],
            'doc_type': doc_timeline[0]['doc_type'],
            'explanation': "First document in sequence"
        })
        
        # Identify last document
        key_documents.append({
            'event_id': doc_timeline[-1]['event_id'],
            'type': 'last_document',
            'date': doc_timeline[-1]['date'],
            'doc_id': doc_timeline[-1]['doc_id'],
            'doc_type': doc_timeline[-1]['doc_type'],
            'explanation': "Last document in sequence"
        })
        
        # Look for significant events
        for i in range(1, len(doc_timeline)):
            prev_doc = doc_timeline[i-1]
            curr_doc = doc_timeline[i]
            
            # Check for document type changes
            if prev_doc['doc_type'] != curr_doc['doc_type']:
                key_documents.append({
                    'event_id': curr_doc['event_id'],
                    'type': 'document_type_change',
                    'date': curr_doc['date'],
                    'doc_id': curr_doc['doc_id'],
                    'from_doc_type': prev_doc['doc_type'],
                    'to_doc_type': curr_doc['doc_type'],
                    'explanation': f"Change in document type from {prev_doc['doc_type']} to {curr_doc['doc_type']}"
                })
            
            # Check for main document
            if curr_doc.get('is_main_document'):
                key_documents.append({
                    'event_id': curr_doc['event_id'],
                    'type': 'main_document',
                    'date': curr_doc['date'],
                    'doc_id': curr_doc['doc_id'],
                    'doc_type': curr_doc['doc_type'],
                    'explanation': "Main document of interest"
                })
            
            # Check for significant gaps
            prev_date = None
            curr_date = None
            
            if prev_doc['date']:
                try:
                    prev_date = datetime.fromisoformat(prev_doc['date'].replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    prev_date = None
                    
            if curr_doc['date']:
                try:
                    curr_date = datetime.fromisoformat(curr_doc['date'].replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    curr_date = None
            
            if prev_date and curr_date:
                gap_days = (curr_date - prev_date).days
                if gap_days > self.significant_gap_days:
                    key_documents.append({
                        'event_id': curr_doc['event_id'],
                        'type': 'significant_gap',
                        'date': curr_doc['date'],
                        'doc_id': curr_doc['doc_id'],
                        'previous_date': prev_doc['date'],
                        'gap_days': gap_days,
                        'explanation': f"Significant gap of {gap_days} days since previous document"
                    })
        
        return key_documents
    
    def _generate_document_sequence(self, doc_timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a sequence analysis of documents.
        
        Args:
            doc_timeline: Document timeline
            
        Returns:
            Document sequence analysis
        """
        # Skip if timeline is empty
        if not doc_timeline:
            return {'sequence': [], 'patterns': []}
            
        # Extract sequence of document types
        sequence = []
        for doc in doc_timeline:
            if doc['date'] and doc['doc_type']:
                sequence.append({
                    'doc_id': doc['doc_id'],
                    'doc_type': doc['doc_type'],
                    'date': doc['date']
                })
        
        # Analyze patterns in the sequence
        patterns = self._analyze_sequence_patterns(sequence)
        
        return {
            'sequence': sequence,
            'patterns': patterns
        }
    
    def _analyze_sequence_patterns(self, sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze patterns in a document sequence.
        
        Args:
            sequence: Document sequence
            
        Returns:
            List of sequence patterns
        """
        patterns = []
        
        # Skip if sequence is too short
        if len(sequence) < 3:
            return patterns
            
        # Check for common document type sequences
        doc_types = [doc['doc_type'] for doc in sequence]
        
        # Look for change order -> approval -> payment application pattern
        for i in range(len(doc_types) - 2):
            if (doc_types[i] == 'change_order' and 
                doc_types[i+1] in ['approval', 'change_order_approval'] and 
                doc_types[i+2] == 'payment_application'):
                    
                patterns.append({
                    'type': 'change_order_payment_sequence',
                    'start_index': i,
                    'end_index': i + 2,
                    'doc_ids': [sequence[i]['doc_id'], sequence[i+1]['doc_id'], sequence[i+2]['doc_id']],
                    'explanation': "Standard change order approval and payment sequence"
                })
        
        # Look for request -> rejection -> re-submission pattern
        for i in range(len(doc_types) - 2):
            if (doc_types[i] in ['request', 'change_order', 'payment_application'] and 
                doc_types[i+1] in ['rejection', 'disapproval'] and 
                doc_types[i+2] in ['request', 'change_order', 'payment_application']):
                    
                patterns.append({
                    'type': 'rejection_resubmission_sequence',
                    'start_index': i,
                    'end_index': i + 2,
                    'doc_ids': [sequence[i]['doc_id'], sequence[i+1]['doc_id'], sequence[i+2]['doc_id']],
                    'explanation': "Request, rejection, and resubmission sequence"
                })
        
        return patterns
    
    def _identify_key_periods(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key periods in a project timeline.
        
        Args:
            timeline: Project timeline
            
        Returns:
            List of key periods
        """
        key_periods = []
        
        # Skip if timeline is empty
        if not timeline or len(timeline) < 2:
            return key_periods
            
        # Sort timeline by date
        sorted_timeline = sorted(
            timeline, 
            key=lambda x: datetime.fromisoformat(x['date'].replace('Z', '+00:00')) if x['date'] else datetime.max
        )
        
        # Define the project start and end
        start_date = datetime.fromisoformat(sorted_timeline[0]['date'].replace('Z', '+00:00')) if sorted_timeline[0]['date'] else None
        end_date = datetime.fromisoformat(sorted_timeline[-1]['date'].replace('Z', '+00:00')) if sorted_timeline[-1]['date'] else None
        
        if start_date and end_date:
            # Add project period
            key_periods.append({
                'type': 'project_duration',
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'duration_days': (end_date - start_date).days,
                'explanation': "Overall project duration"
            })
            
            # Look for periods of high activity
            monthly_counts = {}
            
            for event in sorted_timeline:
                if event['date']:
                    date = datetime.fromisoformat(event['date'].replace('Z', '+00:00'))
                    month_key = date.strftime('%Y-%m')
                    
                    if month_key not in monthly_counts:
                        monthly_counts[month_key] = 0
                        
                    monthly_counts[month_key] += 1
            
            # Calculate average monthly activity
            avg_monthly = sum(monthly_counts.values()) / len(monthly_counts) if monthly_counts else 0
            
            # Identify high activity months
            high_activity_months = []
            for month, count in monthly_counts.items():
                if count > avg_monthly * 1.5:  # 50% above average
                    high_activity_months.append((month, count))
            
            # Add high activity periods
            for month, count in high_activity_months:
                year, month_num = month.split('-')
                month_start = datetime(int(year), int(month_num), 1)
                
                # Calculate month end (handle December specially)
                if int(month_num) == 12:
                    month_end = datetime(int(year) + 1, 1, 1) - timedelta(days=1)
                else:
                    month_end = datetime(int(year), int(month_num) + 1, 1) - timedelta(days=1)
                
                key_periods.append({
                    'type': 'high_activity_period',
                    'start_date': month_start.isoformat(),
                    'end_date': month_end.isoformat(),
                    'duration_days': (month_end - month_start).days + 1,
                    'document_count': count,
                    'activity_ratio': count / avg_monthly,
                    'explanation': f"Period of high activity ({count} documents, {count / avg_monthly:.1f}x average)"
                })
        
        return key_periods
    
    def _generate_project_timeline_summary(self, timeline: List[Dict[str, Any]], 
                                      key_periods: List[Dict[str, Any]]) -> str:
        """Generate a summary of the project timeline.
        
        Args:
            timeline: Project timeline
            key_periods: List of key periods
            
        Returns:
            Project timeline summary
        """
        if not timeline:
            return "No project timeline events available."
            
        # Get project duration
        project_duration = None
        for period in key_periods:
            if period['type'] == 'project_duration':
                project_duration = period
                break
        
        if not project_duration:
            return f"Project includes {len(timeline)} documents with no clear timeline."
        
        # Format date range
        start_date = project_duration['start_date'].split('T')[0]  # Just get the date part
        end_date = project_duration['end_date'].split('T')[0]  # Just get the date part
        duration_days = project_duration['duration_days']
        
        # Count document types
        doc_types = {}
        for event in timeline:
            doc_type = event.get('doc_type')
            if doc_type:
                if doc_type not in doc_types:
                    doc_types[doc_type] = 0
                doc_types[doc_type] += 1
        
        # Format document types
        doc_type_summary = ", ".join(f"{count} {doc_type}" for doc_type, count in doc_types.items())
        
        # Build summary
        summary = f"Project timeline spans from {start_date} to {end_date} ({duration_days} days), with {len(timeline)} documents"
        
        if doc_type_summary:
            summary += f": {doc_type_summary}"
            
        # Add high activity periods
        high_activity_periods = [period for period in key_periods if period['type'] == 'high_activity_period']
        if high_activity_periods:
            summary += f". {len(high_activity_periods)} periods of high activity identified."
        
        return summary