"""
Report generator for the Construction Document Analysis System.

This module provides the core functionality for generating different types of
reports from financial analysis results, including summaries, detailed reports,
and evidence chains.
"""

import os
import json
import logging
import tempfile
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import markdown
import jinja2
from sqlalchemy.orm import Session

# Import WeasyPrint for PDF generation
try:
    from weasyprint import HTML, CSS
    HAS_WEASYPRINT = True
except ImportError:
    HAS_WEASYPRINT = False
    logging.warning("WeasyPrint not installed. PDF generation will be limited to markdown files.")

# Import Excel exporter
try:
    from cdas.reporting.excel_exporter import ExcelExporter
    HAS_EXCEL_EXPORTER = True
except ImportError:
    HAS_EXCEL_EXPORTER = False
    logging.warning("Excel exporter not available. Excel export will be limited.")

from cdas.db.models import Report, Document, LineItem, AnalysisFlag
from cdas.db.operations import create_report, add_report_evidence
from cdas.financial_analysis.engine import FinancialAnalysisEngine
from cdas.financial_analysis.reporting.evidence import EvidenceAssembler
from cdas.financial_analysis.reporting.narrative import NarrativeGenerator

# Set up logging
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates reports from financial analysis data."""
    
    def __init__(self, db_session: Session, config: Optional[Dict[str, Any]] = None):
        """Initialize the report generator.
        
        Args:
            db_session: Database session
            config: Optional configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        self.template_dir = self._get_template_dir()
        self.jinja_env = self._setup_jinja_env()
        
        # Initialize supporting components
        self.analysis_engine = FinancialAnalysisEngine(db_session)
        
        # Configure evidence assembler with screenshot support
        evidence_config = config.copy() if config else {}
        evidence_config.setdefault('include_screenshots', True)
        evidence_config.setdefault('screenshot_dir', str(Path(__file__).parent / 'screenshots'))
        self.evidence_assembler = EvidenceAssembler(db_session, evidence_config)
        
        self.narrative_generator = NarrativeGenerator(db_session)
    
    def _get_template_dir(self) -> Path:
        """Get the template directory path.
        
        Returns:
            Path to the template directory
        """
        # Use template_dir from config if available
        if self.config and 'template_dir' in self.config:
            return Path(self.config['template_dir'])
        
        # Default to the templates directory in the package
        module_dir = Path(__file__).parent
        return module_dir / 'templates'
    
    def _setup_jinja_env(self) -> jinja2.Environment:
        """Set up the Jinja2 template environment.
        
        Returns:
            Jinja2 Environment
        """
        # Create template loader
        if not self.template_dir.exists():
            self.template_dir.mkdir(parents=True, exist_ok=True)
            
        loader = jinja2.FileSystemLoader(str(self.template_dir))
        
        # Create Jinja environment
        env = jinja2.Environment(
            loader=loader,
            autoescape=jinja2.select_autoescape(['html']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        env.filters['currency'] = lambda value: f"${float(value):,.2f}" if value else "$0.00"
        env.filters['date'] = lambda dt: dt.strftime('%Y-%m-%d') if dt else ''
        
        return env
    
    def generate_summary_report(
        self, 
        output_path: str,
        project_id: Optional[str] = None,
        format: str = 'pdf',
        include_evidence: bool = False,
        created_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a summary report.
        
        Args:
            output_path: Path to save the report
            project_id: Project ID to filter data
            format: Output format (pdf, html, md, excel)
            include_evidence: Whether to include evidence
            created_by: User who created the report
            
        Returns:
            Dictionary with report information
        """
        logger.info(f"Generating summary report in {format} format")
        
        # Gather data
        project_data = self._gather_project_data(project_id)
        disputed_amounts = self.analysis_engine.find_disputed_amounts()
        suspicious_patterns = self.analysis_engine.find_suspicious_patterns()
        
        # Generate narrative
        narrative = self.narrative_generator.generate_narrative(
            project_data,
            disputed_amounts,
            suspicious_patterns
        )
        
        # Generate report content based on format
        if format == 'md' or format == 'html':
            md_content = self._generate_markdown_report('summary', {
                'title': narrative['title'],
                'date': datetime.now().strftime('%Y-%m-%d'),
                'project': project_data.get('project_name', 'Unknown Project'),
                'sections': narrative['sections'],
                'disputed_amounts': disputed_amounts,
                'suspicious_patterns': suspicious_patterns,
                'include_evidence': include_evidence
            })
            
            if format == 'html':
                content = markdown.markdown(
                    md_content, 
                    extensions=['tables', 'fenced_code']
                )
            else:
                content = md_content
        else:
            # For Excel, we'll use the same markdown content for now
            # For PDF, use weasyprint to convert HTML to PDF if available
            md_content = self._generate_markdown_report('summary', {
                'title': narrative['title'],
                'date': datetime.now().strftime('%Y-%m-%d'),
                'project': project_data.get('project_name', 'Unknown Project'),
                'sections': narrative['sections'],
                'disputed_amounts': disputed_amounts,
                'suspicious_patterns': suspicious_patterns,
                'include_evidence': include_evidence
            })
            content = md_content
        
        # Process based on format
        if format == 'pdf' and HAS_WEASYPRINT:
            content = self._generate_pdf(
                md_content, 
                output_path, 
                title=narrative['title']
            )
        elif format == 'excel':
            if HAS_EXCEL_EXPORTER:
                # Use Excel exporter
                excel_exporter = ExcelExporter(self.db_session)
                excel_result = excel_exporter.export_financial_summary(output_path, project_id)
                content = json.dumps(excel_result, indent=2)  # Store result as JSON string
            else:
                # Fallback to markdown content
                content = md_content
                with open(output_path, 'w') as f:
                    f.write(content)
        else:
            # Default to writing markdown/HTML directly
            content = md_content
            with open(output_path, 'w') as f:
                f.write(content)
        
        # Save the report in the database
        report = create_report(
            self.db_session,
            title=f"Summary Report - {project_data.get('project_name', 'Unknown Project')}",
            description="Financial analysis summary report",
            content=content,
            format=format,
            created_by=created_by,
            parameters={
                'project_id': project_id,
                'include_evidence': include_evidence,
                'report_type': 'summary'
            }
        )
        
        # Add evidence to the report if requested
        if include_evidence:
            evidence_collection = self.evidence_assembler.assemble_evidence(
                suspicious_patterns + disputed_amounts
            )
            self.evidence_assembler.store_evidence_for_report(
                report.report_id,
                suspicious_patterns + disputed_amounts
            )
        
        return {
            'report_id': report.report_id,
            'title': report.title,
            'format': format,
            'path': output_path,
            'content_length': len(content)
        }
    
    def generate_detailed_report(
        self,
        output_path: str,
        project_id: Optional[str] = None,
        format: str = 'pdf',
        include_evidence: bool = True,
        created_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a detailed report.
        
        Args:
            output_path: Path to save the report
            project_id: Project ID to filter data
            format: Output format (pdf, html, md, excel)
            include_evidence: Whether to include evidence
            created_by: User who created the report
            
        Returns:
            Dictionary with report information
        """
        logger.info(f"Generating detailed report in {format} format")
        
        # Gather data
        project_data = self._gather_project_data(project_id)
        disputed_amounts = self.analysis_engine.find_disputed_amounts()
        suspicious_patterns = self.analysis_engine.find_suspicious_patterns()
        anomalies = self.analysis_engine.find_anomalies()
        
        # Get all document relationships for the project
        relationships = self._gather_document_relationships(project_id)
        
        # Group suspicious patterns by type
        patterns_by_type = {}
        for pattern in suspicious_patterns:
            pattern_type = pattern.get('pattern_type', 'unknown')
            if pattern_type not in patterns_by_type:
                patterns_by_type[pattern_type] = []
            patterns_by_type[pattern_type].append(pattern)
        
        # Generate evidence if requested
        evidence_collection = None
        if include_evidence:
            evidence_collection = self.evidence_assembler.assemble_evidence(
                suspicious_patterns + disputed_amounts
            )
            
            # Add screenshots to evidence if available
            if evidence_collection and hasattr(self.evidence_assembler, 'get_evidence_screenshots'):
                screenshots = self.evidence_assembler.get_evidence_screenshots(evidence_collection)
                
                # Add screenshots to evidence items
                for evidence_id, screenshot_data in screenshots.items():
                    if evidence_id in evidence_collection['evidence_items']:
                        evidence_collection['evidence_items'][evidence_id]['screenshots'] = screenshot_data.get('screenshots', [])
        
        # Generate report content based on format
        if format == 'md' or format == 'html':
            md_content = self._generate_markdown_report('detailed', {
                'title': f"Detailed Financial Analysis Report",
                'date': datetime.now().strftime('%Y-%m-%d'),
                'project': project_data.get('project_name', 'Unknown Project'),
                'project_data': project_data,
                'disputed_amounts': disputed_amounts,
                'suspicious_patterns': suspicious_patterns,
                'patterns_by_type': patterns_by_type,
                'anomalies': anomalies,
                'relationships': relationships,
                'evidence': evidence_collection,
                'include_evidence': include_evidence
            })
            
            if format == 'html':
                content = markdown.markdown(
                    md_content, 
                    extensions=['tables', 'fenced_code']
                )
            else:
                content = md_content
        else:
            # For Excel, we'll use the same markdown content for now
            # For PDF, we'll generate in the save step
            md_content = self._generate_markdown_report('detailed', {
                'title': f"Detailed Financial Analysis Report",
                'date': datetime.now().strftime('%Y-%m-%d'),
                'project': project_data.get('project_name', 'Unknown Project'),
                'project_data': project_data,
                'disputed_amounts': disputed_amounts,
                'suspicious_patterns': suspicious_patterns,
                'patterns_by_type': patterns_by_type,
                'anomalies': anomalies,
                'relationships': relationships,
                'evidence': evidence_collection,
                'include_evidence': include_evidence
            })
            content = md_content
        
        # Process based on format
        if format == 'pdf' and HAS_WEASYPRINT:
            content = self._generate_pdf(
                md_content, 
                output_path, 
                title=f"Detailed Financial Analysis Report"
            )
        elif format == 'excel':
            if HAS_EXCEL_EXPORTER:
                # Use Excel exporter for analysis results
                excel_exporter = ExcelExporter(self.db_session)
                excel_result = excel_exporter.export_analysis_results(output_path, project_id)
                content = json.dumps(excel_result, indent=2)  # Store result as JSON string
            else:
                # Fallback to markdown content
                content = md_content
                with open(output_path, 'w') as f:
                    f.write(content)
        else:
            # Default to writing markdown/HTML directly
            content = md_content
            with open(output_path, 'w') as f:
                f.write(content)
        
        # Save the report in the database
        report = create_report(
            self.db_session,
            title=f"Detailed Report - {project_data.get('project_name', 'Unknown Project')}",
            description="Detailed financial analysis report",
            content=content,
            format=format,
            created_by=created_by,
            parameters={
                'project_id': project_id,
                'include_evidence': include_evidence,
                'report_type': 'detailed'
            }
        )
        
        # Add evidence to the report if include_evidence
        if include_evidence:
            self.evidence_assembler.store_evidence_for_report(
                report.report_id,
                suspicious_patterns + disputed_amounts
            )
        
        return {
            'report_id': report.report_id,
            'title': report.title,
            'format': format,
            'path': output_path,
            'content_length': len(content)
        }
    
    def generate_evidence_report(
        self,
        amount: float,
        output_path: str,
        format: str = 'pdf',
        created_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate an evidence report for a specific amount.
        
        Args:
            amount: Amount to analyze
            output_path: Path to save the report
            format: Output format (pdf, html, md, excel)
            created_by: User who created the report
            
        Returns:
            Dictionary with report information
        """
        logger.info(f"Generating evidence report for amount ${amount:,.2f}")
        
        # Analyze the amount
        analysis_results = self.analysis_engine.analyze_amount(amount)
        matches = analysis_results.get('matches', [])
        anomalies = analysis_results.get('anomalies', [])
        
        # Generate narrative for the amount
        narrative = self.narrative_generator.generate_summary_for_amount(
            amount,
            matches
        )
        
        # Assemble evidence
        evidence_data = []
        for match in matches:
            if 'item_id' in match:
                evidence_data.append({
                    'item_id': match['item_id'],
                    'explanation': f"Amount ${amount:,.2f} found in this line item"
                })
        
        evidence_collection = self.evidence_assembler.assemble_evidence(evidence_data)
        
        # Add screenshots to evidence if available
        if evidence_collection and hasattr(self.evidence_assembler, 'get_evidence_screenshots'):
            screenshots = self.evidence_assembler.get_evidence_screenshots(evidence_collection)
            
            # Add screenshots to evidence items
            for evidence_id, screenshot_data in screenshots.items():
                if evidence_id in evidence_collection['evidence_items']:
                    evidence_collection['evidence_items'][evidence_id]['screenshots'] = screenshot_data.get('screenshots', [])
        
        # Generate report content based on format
        if format == 'md' or format == 'html':
            md_content = self._generate_markdown_report('evidence', {
                'title': f"Evidence Report for Amount ${amount:,.2f}",
                'date': datetime.now().strftime('%Y-%m-%d'),
                'amount': amount,
                'matches': matches,
                'anomalies': anomalies,
                'narrative': narrative,
                'evidence': evidence_collection
            })
            
            if format == 'html':
                content = markdown.markdown(
                    md_content, 
                    extensions=['tables', 'fenced_code']
                )
            else:
                content = md_content
        else:
            # For Excel, we'll use the same markdown content for now
            # For PDF, we'll generate in the save step
            md_content = self._generate_markdown_report('evidence', {
                'title': f"Evidence Report for Amount ${amount:,.2f}",
                'date': datetime.now().strftime('%Y-%m-%d'),
                'amount': amount,
                'matches': matches,
                'anomalies': anomalies,
                'narrative': narrative,
                'evidence': evidence_collection
            })
            content = md_content
        
        # Process based on format
        if format == 'pdf' and HAS_WEASYPRINT:
            content = self._generate_pdf(
                md_content, 
                output_path, 
                title=f"Evidence Report for Amount ${amount:,.2f}"
            )
        elif format == 'excel':
            if HAS_EXCEL_EXPORTER:
                # Use Excel exporter for amount analysis
                excel_exporter = ExcelExporter(self.db_session)
                excel_result = excel_exporter.export_amount_analysis(amount, output_path)
                content = json.dumps(excel_result, indent=2)  # Store result as JSON string
            else:
                # Fallback to markdown content
                content = md_content
                with open(output_path, 'w') as f:
                    f.write(content)
        else:
            # Default to writing markdown/HTML directly
            content = md_content
            with open(output_path, 'w') as f:
                f.write(content)
        
        # Save the report in the database
        report = create_report(
            self.db_session,
            title=f"Evidence Report - ${amount:,.2f}",
            description=f"Evidence report for amount ${amount:,.2f}",
            content=content,
            format=format,
            created_by=created_by,
            parameters={
                'amount': amount,
                'report_type': 'evidence'
            }
        )
        
        # Add evidence to the report
        for match in matches:
            if 'item_id' in match and 'doc_id' in match:
                add_report_evidence(
                    self.db_session,
                    report.report_id,
                    doc_id=match['doc_id'],
                    item_id=match['item_id'],
                    citation_text=f"Amount ${amount:,.2f} found in document {match.get('doc_type', 'Unknown')}",
                    relevance_score=1.0
                )
        
        return {
            'report_id': report.report_id,
            'title': report.title,
            'format': format,
            'path': output_path,
            'content_length': len(content)
        }
    
    def _generate_markdown_report(self, report_type: str, context: Dict[str, Any]) -> str:
        """Generate markdown report content from a template.
        
        Args:
            report_type: Type of report (summary, detailed, evidence)
            context: Template context
            
        Returns:
            Markdown content
        """
        # Look for the template file
        template_name = f"{report_type}.md.j2"
        template_path = self.template_dir / template_name
        
        # If template doesn't exist, create a default one
        if not template_path.exists():
            default_template = self._get_default_template(report_type)
            self.template_dir.mkdir(parents=True, exist_ok=True)
            with open(template_path, 'w') as f:
                f.write(default_template)
        
        # Render the template
        template = self.jinja_env.get_template(template_name)
        return template.render(**context)
    
    def _get_default_template(self, report_type: str) -> str:
        """Get a default template for a report type.
        
        Args:
            report_type: Type of report (summary, detailed, evidence)
            
        Returns:
            Default template content
        """
        if report_type == 'summary':
            return """# {{ title }}

*Generated on: {{ date }}*

## Project: {{ project }}

{% for section in sections %}
{{ "#" * (section.level + 1) }} {{ section.heading }}

{{ section.content }}

{% endfor %}

{% if disputed_amounts %}
## Disputed Amounts

| Amount | Description |
|--------|-------------|
{% for amount in disputed_amounts %}
| {{ amount.amount|currency }} | {{ amount.description }} |
{% endfor %}
{% endif %}

{% if suspicious_patterns %}
## Suspicious Patterns

{% for pattern in suspicious_patterns %}
### {{ pattern.pattern_type|replace('_', ' ')|title }}

{{ pattern.explanation }}

**Confidence:** {{ pattern.confidence * 100 }}%

{% endfor %}
{% endif %}

{% if include_evidence %}
## Evidence

{% for item in disputed_amounts + suspicious_patterns %}
### {{ item.pattern_type|replace('_', ' ')|title if item.pattern_type else 'Disputed Amount' }}

{{ item.explanation }}

**Documents:**
{% for doc_id in item.doc_ids|default([]) %}
- {{ doc_id }}
{% endfor %}

{% endfor %}
{% endif %}
"""
        elif report_type == 'detailed':
            return """# {{ title }}

*Generated on: {{ date }}*

## Project: {{ project }}

{% if project_data %}
**Total Documents:** {{ project_data.total_documents }}
**Total Financial Items:** {{ project_data.total_financial_items }}
**Total Amount Disputed:** {{ project_data.total_amount_disputed|currency }}
{% endif %}

## Disputed Amounts

{% if disputed_amounts %}
| Amount | Description | Document | Party |
|--------|-------------|----------|-------|
{% for amount in disputed_amounts %}
| {{ amount.amount|currency }} | {{ amount.description }} | {{ amount.doc_id }} | {{ amount.party|default('Unknown') }} |
{% endfor %}
{% else %}
No disputed amounts were identified.
{% endif %}

## Suspicious Patterns

{% if patterns_by_type %}
{% for pattern_type, patterns in patterns_by_type.items() %}
### {{ pattern_type|replace('_', ' ')|title }} ({{ patterns|length }})

{% for pattern in patterns %}
#### Pattern {{ loop.index }}

{{ pattern.explanation }}

**Confidence:** {{ pattern.confidence * 100 }}%

**Documents involved:**
{% for doc_id in pattern.doc_ids|default([]) %}
- {{ doc_id }}
{% endfor %}

{% endfor %}
{% endfor %}
{% else %}
No suspicious patterns were identified.
{% endif %}

{% if anomalies %}
## Financial Anomalies

{% for anomaly in anomalies %}
### {{ anomaly.anomaly_type|replace('_', ' ')|title }}

{{ anomaly.explanation }}

**Confidence:** {{ anomaly.confidence * 100 }}%

{% endfor %}
{% else %}
No financial anomalies were detected.
{% endif %}

{% if relationships %}
## Document Relationships

{% for relationship in relationships %}
- {{ relationship.source_doc_id }} -> {{ relationship.target_doc_id }} ({{ relationship.relationship_type|replace('_', ' ')|title }})
{% endfor %}
{% endif %}

{% if include_evidence and evidence %}
## Evidence

**Evidence Count:** {{ evidence.evidence_count }}

{% for evidence_id, evidence_item in evidence.evidence_items.items() %}
### {{ evidence_id }}: {{ evidence_item.type|replace('_', ' ')|title }}

{{ evidence_item.summary }}

**Documents:**
{% for doc in evidence_item.documents %}
- {{ doc.doc_type|default('Unknown') }}: {{ doc.file_name }} ({{ doc.date }})
{% endfor %}

**Line Items:**
{% for item in evidence_item.items %}
- {{ item.description }}: {{ item.amount|currency if item.amount != None }}
{% endfor %}

{% endfor %}
{% endif %}
"""
        elif report_type == 'evidence':
            return """# {{ title }}

*Generated on: {{ date }}*

## Amount: {{ amount|currency }}

{{ narrative }}

## Matches

{% if matches %}
| Document | Description | Amount | Date |
|----------|-------------|--------|------|
{% for match in matches %}
| {{ match.doc_id }} | {{ match.description }} | {{ match.amount|currency }} | {{ match.date|default('Unknown') }} |
{% endfor %}
{% else %}
No matches found for this amount.
{% endif %}

{% if anomalies %}
## Anomalies

{% for anomaly in anomalies %}
### {{ anomaly.anomaly_type|replace('_', ' ')|title }}

{{ anomaly.explanation }}

**Confidence:** {{ anomaly.confidence * 100 }}%

{% endfor %}
{% endif %}

{% if evidence %}
## Evidence

**Evidence Count:** {{ evidence.evidence_count }}

{% for evidence_id, evidence_item in evidence.evidence_items.items() %}
### {{ evidence_id }}

{% for doc in evidence_item.documents %}
**Document:** {{ doc.file_name }}
**Type:** {{ doc.doc_type|default('Unknown') }}
**Date:** {{ doc.date|default('Unknown') }}
{% endfor %}

{% for item in evidence_item.items %}
**Line Item:** {{ item.description }}
**Amount:** {{ item.amount|currency if item.amount != None }}
{% endfor %}

{% endfor %}
{% endif %}
"""
        else:
            return "# {{ title }}\n\n*Generated on: {{ date }}*\n\nNo template available for this report type."
    
    def _gather_project_data(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Gather project data for reporting.
        
        Args:
            project_id: Project ID to filter data
            
        Returns:
            Dictionary with project data
        """
        from cdas.db.operations import get_documents, get_line_items
        
        # Get documents
        documents = get_documents(
            self.db_session,
            project_id=project_id
        )
        
        # Get line items
        line_items = get_line_items(
            self.db_session,
            project_id=project_id
        )
        
        # Calculate summary statistics
        total_documents = len(documents)
        total_items = len(line_items)
        
        # Calculate total amount (excluding non-financial documents)
        total_amount = sum(
            float(item.amount) 
            for item in line_items 
            if item.amount is not None
        )
        
        # Calculate disputed amount (simplistic approach for now)
        disputed_amounts = self.analysis_engine.find_disputed_amounts()
        total_disputed = sum(
            amount.get('amount', 0) 
            for amount in disputed_amounts 
            if amount.get('amount') is not None
        )
        
        # Get project name (simplistic approach)
        project_name = "Unknown Project"
        if project_id:
            project_name = f"Project {project_id}"
        elif total_documents > 0:
            # Try to extract project name from the first document
            doc = documents[0]
            if hasattr(doc, 'meta_data') and doc.meta_data:
                project_name = doc.meta_data.get('project_name', project_name)
        
        return {
            'project_id': project_id,
            'project_name': project_name,
            'total_documents': total_documents,
            'total_financial_items': total_items,
            'total_amount': total_amount,
            'total_amount_disputed': total_disputed,
            'document_count_by_type': self._count_documents_by_type(documents),
            'document_count_by_party': self._count_documents_by_party(documents),
            'generated_at': datetime.now().isoformat()
        }
    
    def _count_documents_by_type(self, documents: List[Document]) -> Dict[str, int]:
        """Count documents by type.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary mapping document types to counts
        """
        counts = {}
        for doc in documents:
            doc_type = doc.doc_type or 'unknown'
            if doc_type not in counts:
                counts[doc_type] = 0
            counts[doc_type] += 1
        return counts
    
    def _count_documents_by_party(self, documents: List[Document]) -> Dict[str, int]:
        """Count documents by party.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary mapping parties to counts
        """
        counts = {}
        for doc in documents:
            party = doc.party or 'unknown'
            if party not in counts:
                counts[party] = 0
            counts[party] += 1
        return counts
    
    def _gather_document_relationships(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Gather document relationships.
        
        Args:
            project_id: Project ID to filter data
            
        Returns:
            List of relationship dictionaries
        """
        from cdas.db.models import DocumentRelationship
        
        # Query all relationships
        relationships_query = self.db_session.query(DocumentRelationship)
        
        # If project_id is provided, filter by documents in the project
        # This is a simplistic approach - in a real implementation, you'd want to filter
        # more efficiently using JOIN operations
        if project_id:
            from cdas.db.operations import get_documents
            documents = get_documents(self.db_session, project_id=project_id)
            doc_ids = [doc.doc_id for doc in documents]
            
            relationships_query = relationships_query.filter(
                DocumentRelationship.source_doc_id.in_(doc_ids) |
                DocumentRelationship.target_doc_id.in_(doc_ids)
            )
        
        # Convert to dictionaries
        relationships = []
        for rel in relationships_query.all():
            relationships.append({
                'source_doc_id': rel.source_doc_id,
                'target_doc_id': rel.target_doc_id,
                'relationship_type': rel.relationship_type,
                'confidence': float(rel.confidence) if rel.confidence else 1.0
            })
        
        return relationships
        
    def _generate_pdf(self, markdown_content: str, output_path: str, title: str = None) -> str:
        """Generate a PDF file from markdown content.
        
        Args:
            markdown_content: Markdown content to convert to PDF
            output_path: Path to save the PDF file
            title: Title of the PDF document
            
        Returns:
            The HTML content used to generate the PDF
        """
        if not HAS_WEASYPRINT:
            logger.warning("WeasyPrint not installed. Saving as markdown with .pdf extension.")
            with open(output_path, 'w') as f:
                f.write(markdown_content)
            return markdown_content
            
        logger.info(f"Generating PDF: {output_path}")
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            markdown_content,
            extensions=['tables', 'fenced_code']
        )
        
        # Get CSS file path
        css_path = self.template_dir / 'report.css'
        
        # Ensure the CSS file exists
        if not css_path.exists():
            # Create default CSS if it doesn't exist
            self._create_default_css(css_path)
        
        # Create complete HTML document
        html_doc = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title or 'CDAS Report'}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <div class="header">
        <h1>{title or 'CDAS Report'}</h1>
        <div class="report-info">
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
    </div>
    {html_content}
</body>
</html>"""
        
        # Create temporary HTML file
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            tmp.write(html_doc.encode('utf-8'))
            tmp_html_path = tmp.name
            
        try:
            # Generate PDF from HTML
            HTML(tmp_html_path).write_pdf(
                output_path,
                stylesheets=[CSS(css_path)]
            )
            logger.info(f"PDF generated successfully: {output_path}")
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            # Fallback to markdown if PDF generation fails
            with open(output_path, 'w') as f:
                f.write(markdown_content)
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_html_path)
            except:
                pass
                
        return html_doc
    
    def _create_default_css(self, css_path: Path) -> None:
        """Create a default CSS file for PDF styling.
        
        Args:
            css_path: Path to save the CSS file
        """
        default_css = """/* Default CSS for CDAS Reports */

body {
    font-family: Arial, sans-serif;
    margin: 1in 0.75in;
    font-size: 11pt;
    line-height: 1.5;
}

h1 { font-size: 18pt; color: #000080; }
h2 { font-size: 14pt; color: #000080; border-bottom: 1pt solid #cccccc; }
h3 { font-size: 12pt; color: #000080; }

table {
    width: 100%;
    border-collapse: collapse;
    margin: 10pt 0;
}

th, td {
    border: 1pt solid #ddd;
    padding: 6pt;
    text-align: left;
}

th {
    background-color: #f2f2f2;
}

tr:nth-child(even) {
    background-color: #f9f9f9;
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20pt;
    border-bottom: 1pt solid #cccccc;
    padding-bottom: 10pt;
}

.report-info {
    text-align: right;
    font-size: 9pt;
    color: #666;
}

img {
    max-width: 100%;
}

@page {
    @bottom-center {
        content: "Construction Document Analysis System • Page " counter(page) " of " counter(pages);
        font-size: 9pt;
    }
}
"""
        
        with open(css_path, 'w') as f:
            f.write(default_css)