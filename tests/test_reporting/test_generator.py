"""
Tests for the report generator.

This module contains tests for the report generator functionality.
"""

import os
import tempfile
import pytest
from datetime import datetime
from pathlib import Path

from cdas.reporting.generator import ReportGenerator
from cdas.db.models import Document, LineItem, Report, ReportEvidence
from cdas.db.operations import register_document, store_line_items


@pytest.fixture
def setup_test_data(session):
    """Set up test data for report generation tests."""
    # Create some test documents
    doc1 = register_document(
        session,
        file_path='/path/to/test_doc1.pdf',
        doc_type='payment_app',
        party='contractor',
        date_created=datetime(2023, 1, 15),
        metadata={'project_id': 'test_project', 'project_name': 'Test Project'}
    )
    
    doc2 = register_document(
        session,
        file_path='/path/to/test_doc2.pdf',
        doc_type='change_order',
        party='contractor',
        date_created=datetime(2023, 2, 10),
        metadata={'project_id': 'test_project', 'project_name': 'Test Project'}
    )
    
    doc3 = register_document(
        session,
        file_path='/path/to/test_doc3.pdf',
        doc_type='invoice',
        party='subcontractor',
        date_created=datetime(2023, 2, 15),
        metadata={'project_id': 'test_project', 'project_name': 'Test Project'}
    )
    
    # Create some line items
    items1 = store_line_items(
        session,
        doc1.doc_id,
        [
            {
                'description': 'HVAC Installation',
                'amount': 10000.00,
                'cost_code': 'HVAC-001',
                'category': 'mechanical'
            },
            {
                'description': 'Plumbing Fixtures',
                'amount': 5000.00,
                'cost_code': 'PLUMB-001',
                'category': 'plumbing'
            }
        ]
    )
    
    items2 = store_line_items(
        session,
        doc2.doc_id,
        [
            {
                'description': 'Additional HVAC Work',
                'amount': 2500.00,
                'cost_code': 'HVAC-002',
                'category': 'mechanical'
            }
        ]
    )
    
    items3 = store_line_items(
        session,
        doc3.doc_id,
        [
            {
                'description': 'Electrical Materials',
                'amount': 7500.00,
                'cost_code': 'ELEC-001',
                'category': 'electrical'
            },
            {
                'description': 'HVAC Equipment',
                'amount': 10000.00,  # Duplicate amount
                'cost_code': 'HVAC-001',
                'category': 'mechanical'
            }
        ]
    )
    
    session.commit()
    
    return {
        'documents': [doc1, doc2, doc3],
        'line_items': items1 + items2 + items3
    }


def test_report_generator_initialization(session):
    """Test that the report generator initializes correctly."""
    generator = ReportGenerator(session)
    
    # Check that the generator is initialized with the correct components
    assert generator.db_session == session
    assert generator.template_dir is not None
    assert generator.jinja_env is not None
    assert generator.analysis_engine is not None
    assert generator.evidence_assembler is not None
    assert generator.narrative_generator is not None


def test_generate_summary_report(session, setup_test_data):
    """Test generating a summary report."""
    generator = ReportGenerator(session)
    
    # Create a temporary file for the report
    with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp_file:
        output_path = temp_file.name
    
    try:
        # Generate a summary report
        result = generator.generate_summary_report(
            output_path,
            project_id='test_project',
            format='md',
            include_evidence=False
        )
        
        # Check that the report was generated
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        # Check the returned metadata
        assert result['report_id'] is not None
        assert result['title'] is not None
        assert result['format'] == 'md'
        assert result['path'] == output_path
        assert result['content_length'] > 0
        
        # Check that a report was saved in the database
        report = session.query(Report).filter(Report.report_id == result['report_id']).first()
        assert report is not None
        assert report.title is not None
        assert report.format == 'md'
        assert report.content is not None
        
    finally:
        # Clean up the temporary file
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_generate_detailed_report(session, setup_test_data):
    """Test generating a detailed report."""
    generator = ReportGenerator(session)
    
    # Create a temporary file for the report
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_file:
        output_path = temp_file.name
    
    try:
        # Generate a detailed report
        result = generator.generate_detailed_report(
            output_path,
            project_id='test_project',
            format='html',
            include_evidence=True
        )
        
        # Check that the report was generated
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        # Check the returned metadata
        assert result['report_id'] is not None
        assert result['title'] is not None
        assert result['format'] == 'html'
        assert result['path'] == output_path
        assert result['content_length'] > 0
        
        # Check that a report was saved in the database
        report = session.query(Report).filter(Report.report_id == result['report_id']).first()
        assert report is not None
        assert report.title is not None
        assert report.format == 'html'
        assert report.content is not None
        
        # Check that the HTML content contains the expected format
        with open(output_path, 'r') as f:
            content = f.read()
        assert '<h1>' in content
        assert '<table>' in content
        
        # Check that evidence was added
        evidence = session.query(ReportEvidence).filter(
            ReportEvidence.report_id == result['report_id']
        ).all()
        assert len(evidence) > 0
        
    finally:
        # Clean up the temporary file
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_generate_evidence_report(session, setup_test_data):
    """Test generating an evidence report."""
    generator = ReportGenerator(session)
    
    # Create a temporary file for the report
    with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp_file:
        output_path = temp_file.name
    
    try:
        # Generate an evidence report
        result = generator.generate_evidence_report(
            amount=10000.00,  # Amount that exists in the test data
            output_path=output_path,
            format='md'
        )
        
        # Check that the report was generated
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        # Check the returned metadata
        assert result['report_id'] is not None
        assert result['title'] is not None
        assert result['format'] == 'md'
        assert result['path'] == output_path
        assert result['content_length'] > 0
        
        # Check that a report was saved in the database
        report = session.query(Report).filter(Report.report_id == result['report_id']).first()
        assert report is not None
        assert report.title is not None
        assert report.format == 'md'
        assert report.content is not None
        assert '$10,000.00' in report.content
        
        # Check that evidence was added
        evidence = session.query(ReportEvidence).filter(
            ReportEvidence.report_id == result['report_id']
        ).all()
        assert len(evidence) > 0
        
    finally:
        # Clean up the temporary file
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_template_rendering(session):
    """Test that templates are rendered correctly."""
    generator = ReportGenerator(session)
    
    # Create a temporary template
    template_content = """# Test Template
    
Title: {{ title }}
Date: {{ date }}

## Project: {{ project }}

{% if project_data %}
Total Documents: {{ project_data.total_documents }}
{% endif %}
"""
    
    # Ensure the template directory exists
    os.makedirs(generator.template_dir, exist_ok=True)
    
    template_path = generator.template_dir / 'test.md.j2'
    with open(template_path, 'w') as f:
        f.write(template_content)
    
    try:
        # Render the template
        result = generator._generate_markdown_report('test', {
            'title': 'Test Report',
            'date': '2023-03-01',
            'project': 'Test Project',
            'project_data': {
                'total_documents': 3
            }
        })
        
        # Check that the template was rendered correctly
        assert '# Test Template' in result
        assert 'Title: Test Report' in result
        assert 'Date: 2023-03-01' in result
        assert 'Project: Test Project' in result
        assert 'Total Documents: 3' in result
        
    finally:
        # Clean up the temporary template
        if os.path.exists(template_path):
            os.unlink(template_path)