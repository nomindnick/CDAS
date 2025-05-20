"""
Narrative generator for the financial analysis engine.

This module provides functionality to generate natural language narratives 
explaining financial findings and patterns detected in construction documents.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session


class NarrativeGenerator:
    """Generates natural language narratives for financial findings."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the narrative generator.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
    
    def generate_narrative(self, summary: Dict[str, Any], 
                        disputed_amounts: List[Dict[str, Any]],
                        suspicious_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive narrative from financial findings.
        
        Args:
            summary: Summary statistics
            disputed_amounts: List of disputed amounts
            suspicious_patterns: List of suspicious patterns
            
        Returns:
            Narrative structure
        """
        # Generate individual narrative sections
        summary_narrative = self._generate_summary_narrative(summary)
        disputed_amounts_narrative = self._generate_disputed_amounts_narrative(disputed_amounts)
        suspicious_patterns_narrative = self._generate_suspicious_patterns_narrative(suspicious_patterns)
        
        # Combine into a complete narrative
        narrative = {
            'title': "Financial Analysis Report",
            'generated_at': datetime.utcnow().isoformat(),
            'sections': [
                {
                    'heading': "Executive Summary",
                    'content': summary_narrative,
                    'level': 1
                },
                {
                    'heading': "Disputed Amounts",
                    'content': disputed_amounts_narrative,
                    'level': 1
                },
                {
                    'heading': "Suspicious Patterns",
                    'content': suspicious_patterns_narrative,
                    'level': 1
                }
            ]
        }
        
        return narrative
    
    def _generate_summary_narrative(self, summary: Dict[str, Any]) -> str:
        """Generate a narrative summary of the financial analysis.
        
        Args:
            summary: Summary statistics
            
        Returns:
            Summary narrative
        """
        # For now, just use a placeholder
        if not summary:
            return "No financial summary data available."
        
        total_documents = summary.get('total_documents', 0)
        total_items = summary.get('total_financial_items', 0)
        total_disputed = summary.get('total_amount_disputed', 0)
        
        return f"""
        This report summarizes the analysis of {total_documents} construction documents
        containing {total_items} financial line items. The total amount in dispute is 
        ${total_disputed:,.2f}. Review the detailed sections for specific findings and evidence.
        """
    
    def _generate_disputed_amounts_narrative(self, disputed_amounts: List[Dict[str, Any]]) -> str:
        """Generate a narrative about disputed amounts.
        
        Args:
            disputed_amounts: List of disputed amounts
            
        Returns:
            Disputed amounts narrative
        """
        if not disputed_amounts:
            return "No disputed amounts were identified in the analyzed documents."
        
        # Calculate total disputed amount
        total_disputed = sum(
            amt.get('amount', 0) 
            for amt in disputed_amounts 
            if amt.get('amount') is not None
        )
        
        # Generate summary paragraph
        narrative = f"""
        Analysis identified {len(disputed_amounts)} disputed amounts totaling ${total_disputed:,.2f}.
        These disputed amounts represent discrepancies between contractor claims and district records.
        """
        
        # Add details for each disputed amount
        for i, dispute in enumerate(disputed_amounts):
            amount = dispute.get('amount', 0)
            description = dispute.get('description', 'Unknown item')
            
            narrative += f"\n\n{i+1}. ${amount:,.2f} - {description}"
            
            if dispute.get('explanation'):
                narrative += f"\n   {dispute['explanation']}"
        
        return narrative
    
    def _generate_suspicious_patterns_narrative(self, suspicious_patterns: List[Dict[str, Any]]) -> str:
        """Generate a narrative about suspicious patterns.
        
        Args:
            suspicious_patterns: List of suspicious patterns
            
        Returns:
            Suspicious patterns narrative
        """
        if not suspicious_patterns:
            return "No suspicious patterns were identified in the analyzed documents."
        
        # Generate summary paragraph
        narrative = f"""
        Analysis identified {len(suspicious_patterns)} potentially suspicious patterns
        in the financial records. These patterns may warrant further investigation.
        """
        
        # Group patterns by type
        pattern_types = {}
        for pattern in suspicious_patterns:
            pattern_type = pattern.get('pattern_type', 'Unknown')
            if pattern_type not in pattern_types:
                pattern_types[pattern_type] = []
            pattern_types[pattern_type].append(pattern)
        
        # Generate narrative for each pattern type
        for pattern_type, patterns in pattern_types.items():
            pattern_count = len(patterns)
            narrative += f"\n\n### {pattern_type.replace('_', ' ').title()} ({pattern_count})"
            
            for i, pattern in enumerate(patterns):
                confidence = pattern.get('confidence', 0.0) * 100
                explanation = pattern.get('explanation', 'No explanation provided')
                
                narrative += f"\n\n{i+1}. Confidence: {confidence:.1f}%"
                narrative += f"\n   {explanation}"
        
        return narrative
    
    def generate_summary_for_amount(self, amount: float, 
                                  matches: List[Dict[str, Any]]) -> str:
        """Generate a narrative summary for an amount.
        
        Args:
            amount: Amount to analyze
            matches: List of matches for the amount
            
        Returns:
            Amount summary narrative
        """
        if not matches:
            return f"The amount ${amount:,.2f} was not found in any documents."
        
        # Count occurrences by document type
        doc_types = {}
        parties = {}
        
        for match in matches:
            doc_type = match.get('doc_type', 'Unknown')
            if doc_type not in doc_types:
                doc_types[doc_type] = 0
            doc_types[doc_type] += 1
            
            party = match.get('party', 'Unknown')
            if party not in parties:
                parties[party] = 0
            parties[party] += 1
        
        # Generate summary paragraph
        narrative = f"""
        The amount ${amount:,.2f} appears in {len(matches)} locations across the analyzed documents.
        """
        
        # Add document type breakdown
        narrative += "\n\n### Document Type Breakdown"
        for doc_type, count in doc_types.items():
            narrative += f"\n- {doc_type}: {count} occurrences"
        
        # Add party breakdown
        narrative += "\n\n### Party Breakdown"
        for party, count in parties.items():
            narrative += f"\n- {party}: {count} occurrences"
        
        # Add chronological information if available
        earliest_match = None
        latest_match = None
        
        for match in matches:
            if 'date' in match and match['date']:
                date = datetime.fromisoformat(match['date'].replace('Z', '+00:00'))
                if earliest_match is None or date < earliest_match[0]:
                    earliest_match = (date, match)
                if latest_match is None or date > latest_match[0]:
                    latest_match = (date, match)
        
        if earliest_match and latest_match:
            narrative += "\n\n### Chronology"
            
            if earliest_match[0] == latest_match[0]:
                narrative += f"\nThis amount appears only on {earliest_match[0].strftime('%Y-%m-%d')}."
            else:
                earliest_doc_type = earliest_match[1].get('doc_type', 'Unknown')
                latest_doc_type = latest_match[1].get('doc_type', 'Unknown')
                
                narrative += f"\nFirst appearance: {earliest_match[0].strftime('%Y-%m-%d')} ({earliest_doc_type})"
                narrative += f"\nLatest appearance: {latest_match[0].strftime('%Y-%m-%d')} ({latest_doc_type})"
        
        return narrative
    
    def generate_report_for_document(self, doc_id: str, 
                                   analysis_results: Dict[str, Any]) -> str:
        """Generate a narrative report for a document.
        
        Args:
            doc_id: Document ID
            analysis_results: Analysis results
            
        Returns:
            Document report narrative
        """
        # Extract document information
        patterns = analysis_results.get('patterns', [])
        anomalies = analysis_results.get('anomalies', [])
        matches = analysis_results.get('amount_matches', [])
        relationships = analysis_results.get('relationships', {})
        
        # Generate document overview
        doc_info = relationships.get('document', {})
        doc_type = doc_info.get('doc_type', 'Unknown')
        party = doc_info.get('party', 'Unknown')
        file_name = doc_info.get('file_name', 'Unknown')
        
        narrative = f"""
        # Document Analysis: {file_name}
        
        ## Overview
        
        Document Type: {doc_type}
        Party: {party}
        File: {file_name}
        """
        
        # Add pattern summary
        if patterns:
            narrative += f"\n\n## Patterns ({len(patterns)})"
            for i, pattern in enumerate(patterns):
                pattern_type = pattern.get('pattern_type', 'Unknown')
                confidence = pattern.get('confidence', 0.0) * 100
                explanation = pattern.get('explanation', 'No explanation provided')
                
                narrative += f"\n\n{i+1}. {pattern_type.replace('_', ' ').title()} (Confidence: {confidence:.1f}%)"
                narrative += f"\n   {explanation}"
        else:
            narrative += "\n\n## Patterns\n\nNo patterns were detected in this document."
        
        # Add anomaly summary
        if anomalies:
            narrative += f"\n\n## Anomalies ({len(anomalies)})"
            for i, anomaly in enumerate(anomalies):
                anomaly_type = anomaly.get('anomaly_type', 'Unknown')
                confidence = anomaly.get('confidence', 0.0) * 100
                explanation = anomaly.get('explanation', 'No explanation provided')
                
                narrative += f"\n\n{i+1}. {anomaly_type.replace('_', ' ').title()} (Confidence: {confidence:.1f}%)"
                narrative += f"\n   {explanation}"
        else:
            narrative += "\n\n## Anomalies\n\nNo anomalies were detected in this document."
        
        # Add amount matches summary
        if matches:
            narrative += f"\n\n## Amount Matches ({len(matches)})"
            for i, match in enumerate(matches):
                amount = match.get('amount', 0)
                match_count = match.get('match_count', 0)
                
                narrative += f"\n\n{i+1}. ${amount:,.2f} - Found in {match_count} other documents"
        else:
            narrative += "\n\n## Amount Matches\n\nNo matching amounts were found in other documents."
        
        # Add relationship summary
        if relationships.get('relationships'):
            rel_list = relationships.get('relationships', [])
            narrative += f"\n\n## Document Relationships ({len(rel_list)})"
            
            for i, rel in enumerate(rel_list):
                rel_type = rel.get('relationship_type', 'Unknown')
                rel_dir = rel.get('relationship_direction', 'outgoing')
                confidence = rel.get('confidence', 0.0) * 100
                target_doc = rel.get('related_doc_id', 'Unknown')
                
                narrative += f"\n\n{i+1}. {rel_type.replace('_', ' ').title()} ({rel_dir}, Confidence: {confidence:.1f}%)"
                narrative += f"\n   Related document: {target_doc}"
        else:
            narrative += "\n\n## Document Relationships\n\nNo relationships were detected for this document."
        
        return narrative