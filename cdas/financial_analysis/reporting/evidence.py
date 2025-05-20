"""
Evidence assembler for the financial analysis engine.

This module provides functionality to assemble evidence for findings in the
financial analysis, including document citations, line item references, and
highlighting of relevant information.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from sqlalchemy.orm import Session

from cdas.db.models import Document, LineItem, Page, Annotation, ReportEvidence
from cdas.db.operations import add_report_evidence

# Import screenshot generator
try:
    from cdas.financial_analysis.reporting.screenshots import EvidenceScreenshotGenerator
    HAS_SCREENSHOT_SUPPORT = True
except ImportError:
    HAS_SCREENSHOT_SUPPORT = False


class EvidenceAssembler:
    """Assembles evidence for financial analysis findings."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the evidence assembler.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Initialize screenshot generator if available
        self.screenshot_generator = None
        self.include_screenshots = self.config.get('include_screenshots', False)
        
        if HAS_SCREENSHOT_SUPPORT and self.include_screenshots:
            screenshot_dir = self.config.get('screenshot_dir', None)
            self.screenshot_generator = EvidenceScreenshotGenerator(screenshot_dir)
            self.screenshot_cache = {}
    
    def assemble_evidence(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assemble evidence for a list of findings.
        
        Args:
            findings: List of finding dictionaries
            
        Returns:
            Evidence data structure
        """
        evidence_collection = {}
        
        for i, finding in enumerate(findings):
            evidence_id = f"evidence_{i+1}"
            evidence = self._collect_evidence_for_finding(finding)
            
            if evidence:
                evidence_collection[evidence_id] = evidence
        
        return {
            'evidence_count': len(evidence_collection),
            'evidence_items': evidence_collection,
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _collect_evidence_for_finding(self, finding: Dict[str, Any]) -> Dict[str, Any]:
        """Collect evidence for a specific finding.
        
        Args:
            finding: Finding dictionary
            
        Returns:
            Evidence data structure
        """
        evidence = {
            'summary': finding.get('explanation', 'No explanation provided'),
            'type': finding.get('pattern_type', finding.get('relationship_type', 'general')),
            'confidence': finding.get('confidence', 0.0),
            'items': [],
            'documents': [],
            'citations': []
        }
        
        # Collect documents
        if 'doc_id' in finding:
            document = self._get_document_info(finding['doc_id'])
            if document:
                evidence['documents'].append(document)
        
        if 'source_doc_id' in finding:
            document = self._get_document_info(finding['source_doc_id'])
            if document:
                evidence['documents'].append(document)
        
        if 'target_doc_id' in finding:
            document = self._get_document_info(finding['target_doc_id'])
            if document:
                evidence['documents'].append(document)
        
        # Collect line items
        if 'item_id' in finding:
            item = self._get_line_item_info(finding['item_id'])
            if item:
                evidence['items'].append(item)
        
        if 'source_item_id' in finding:
            item = self._get_line_item_info(finding['source_item_id'])
            if item:
                evidence['items'].append(item)
        
        if 'target_item_id' in finding:
            item = self._get_line_item_info(finding['target_item_id'])
            if item:
                evidence['items'].append(item)
        
        # If there's a list of matches or items
        if 'matches' in finding and isinstance(finding['matches'], list):
            for match in finding['matches']:
                if 'item_id' in match:
                    item = self._get_line_item_info(match['item_id'])
                    if item:
                        evidence['items'].append(item)
                
                if 'doc_id' in match:
                    document = self._get_document_info(match['doc_id'])
                    if document:
                        evidence['documents'].append(document)
        
        # Deduplicate items and documents
        if evidence['items']:
            evidence['items'] = self._deduplicate_by_key(evidence['items'], 'item_id')
        
        if evidence['documents']:
            evidence['documents'] = self._deduplicate_by_key(evidence['documents'], 'doc_id')
        
        # Generate citations
        evidence['citations'] = self._generate_citations(evidence['items'], evidence['documents'])
        
        # Add screenshots if enabled
        if self.include_screenshots and self.screenshot_generator and evidence['items']:
            evidence['screenshots'] = self._get_screenshots_for_items(evidence['items'])
        
        return evidence
    
    def _get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document information dictionary or None if not found
        """
        document = self.db_session.query(Document).filter_by(doc_id=doc_id).first()
        
        if not document:
            return None
        
        return {
            'doc_id': document.doc_id,
            'file_name': document.file_name,
            'doc_type': document.doc_type,
            'party': document.party,
            'date': document.date_created.isoformat() if document.date_created else None,
            'status': document.status
        }
    
    def _get_line_item_info(self, item_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """Get information about a line item.
        
        Args:
            item_id: Line item ID
            
        Returns:
            Line item information dictionary or None if not found
        """
        item = self.db_session.query(LineItem).filter_by(item_id=item_id).first()
        
        if not item:
            return None
        
        return {
            'item_id': item.item_id,
            'doc_id': item.doc_id,
            'description': item.description,
            'amount': float(item.amount) if item.amount is not None else None,
            'cost_code': item.cost_code,
            'category': item.category,
            'status': item.status
        }
    
    def _get_screenshots_for_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get screenshots for line items.
        
        Args:
            items: List of line item dictionaries
            
        Returns:
            List of screenshot dictionaries
        """
        if not self.screenshot_generator:
            return []
        
        screenshots = []
        
        for item in items:
            item_id = item.get('item_id')
            if not item_id:
                continue
                
            # Check if we already have a screenshot for this item
            if item_id in self.screenshot_cache:
                screenshots.append(self.screenshot_cache[item_id])
                continue
            
            # Get the line item object
            line_item = self.db_session.query(LineItem).filter_by(item_id=item_id).first()
            if not line_item:
                continue
            
            # Generate screenshot
            screenshot = self.screenshot_generator.get_screenshot_for_line_item(line_item)
            
            if screenshot.get('success'):
                # Cache the screenshot
                self.screenshot_cache[item_id] = screenshot
                screenshots.append(screenshot)
        
        return screenshots
    
    def _generate_citations(self, items: List[Dict[str, Any]], 
                          documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate citations for evidence.
        
        Args:
            items: List of line item dictionaries
            documents: List of document dictionaries
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        # Group items by document
        items_by_doc = {}
        for item in items:
            doc_id = item.get('doc_id')
            if doc_id:
                if doc_id not in items_by_doc:
                    items_by_doc[doc_id] = []
                items_by_doc[doc_id].append(item)
        
        # Create citations for each document and its items
        for doc in documents:
            doc_id = doc.get('doc_id')
            if not doc_id:
                continue
                
            # Create citation for the document itself
            doc_citation = {
                'type': 'document',
                'doc_id': doc_id,
                'file_name': doc.get('file_name', ''),
                'doc_type': doc.get('doc_type', ''),
                'party': doc.get('party', ''),
                'date': doc.get('date', ''),
                'items': []
            }
            
            # Add items from this document
            if doc_id in items_by_doc:
                for item in items_by_doc[doc_id]:
                    item_citation = {
                        'type': 'line_item',
                        'item_id': item.get('item_id', ''),
                        'description': item.get('description', ''),
                        'amount': item.get('amount', None)
                    }
                    doc_citation['items'].append(item_citation)
            
            citations.append(doc_citation)
        
        return citations
    
    def _deduplicate_by_key(self, items: List[Dict[str, Any]], 
                          key: str) -> List[Dict[str, Any]]:
        """Deduplicate a list of dictionaries by a key.
        
        Args:
            items: List of dictionaries
            key: Key to deduplicate by
            
        Returns:
            Deduplicated list
        """
        seen = set()
        result = []
        
        for item in items:
            item_key = item.get(key)
            if item_key and item_key not in seen:
                seen.add(item_key)
                result.append(item)
        
        return result

    def get_evidence_screenshots(self, evidence_collection: Dict[str, Any]) -> Dict[str, Any]:
        """Get screenshots for evidence collection.
        
        Args:
            evidence_collection: Evidence collection dictionary
            
        Returns:
            Dictionary mapping evidence IDs to screenshots
        """
        if not self.screenshot_generator or not self.include_screenshots:
            return {}
        
        # Extract evidence items
        evidence_items = []
        for evidence_id, evidence in evidence_collection.get('evidence_items', {}).items():
            evidence['evidence_id'] = evidence_id
            evidence_items.append(evidence)
        
        # Get screenshots
        return self.screenshot_generator.get_screenshots_for_evidence(evidence_items)
    
    def store_evidence_for_report(self, report_id: int, 
                                findings: List[Dict[str, Any]]) -> List[ReportEvidence]:
        """Store evidence for a report in the database.
        
        Args:
            report_id: Report ID
            findings: List of finding dictionaries
            
        Returns:
            List of ReportEvidence objects
        """
        evidence_items = []
        
        for finding in findings:
            # Extract document references
            doc_ids = []
            if 'doc_id' in finding:
                doc_ids.append(finding['doc_id'])
            if 'source_doc_id' in finding:
                doc_ids.append(finding['source_doc_id'])
            if 'target_doc_id' in finding:
                doc_ids.append(finding['target_doc_id'])
            
            # Extract item references
            item_ids = []
            if 'item_id' in finding:
                item_ids.append(finding['item_id'])
            if 'source_item_id' in finding:
                item_ids.append(finding['source_item_id'])
            if 'target_item_id' in finding:
                item_ids.append(finding['target_item_id'])
            
            # Create evidence entries
            for doc_id in doc_ids:
                citation_text = f"Document evidence: {finding.get('explanation', '')}"
                evidence = add_report_evidence(
                    self.db_session,
                    report_id,
                    doc_id=doc_id,
                    citation_text=citation_text,
                    relevance_score=finding.get('confidence', 1.0)
                )
                evidence_items.append(evidence)
            
            for item_id in item_ids:
                citation_text = f"Line item evidence: {finding.get('explanation', '')}"
                evidence = add_report_evidence(
                    self.db_session,
                    report_id,
                    item_id=item_id,
                    citation_text=citation_text,
                    relevance_score=finding.get('confidence', 1.0)
                )
                evidence_items.append(evidence)
        
        return evidence_items