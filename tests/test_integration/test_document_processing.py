"""
Integration tests for document processing workflows.

These tests verify the end-to-end functionality of document ingestion, 
extraction, and storage in the database.
"""

import os
import pytest
from pathlib import Path

from cdas.document_processor.factory import DocumentProcessorFactory
from cdas.document_processor.processor import DocumentProcessor
from cdas.db.models import Document, LineItem
from cdas.db.operations import get_document_by_id, get_documents_by_type


def test_document_processor_factory(test_session):
    """Test that the document processor factory creates a valid processor."""
    factory = DocumentProcessorFactory()
    processor = factory.create_processor(test_session)
    
    assert processor is not None
    assert isinstance(processor, DocumentProcessor)
    assert processor.session == test_session


def test_process_existing_document(test_session, sample_payment_app):
    """Test processing an existing document."""
    doc, file_path = sample_payment_app
    
    # Process the document
    factory = DocumentProcessorFactory()
    processor = factory.create_processor(test_session)
    
    result = processor.process_document(
        file_path,
        "payment_app",
        "contractor",
        save_to_db=True
    )
    
    # Verify the result
    assert result is not None
    assert result.success
    assert result.document_id is not None
    assert len(result.extracted_data) > 0
    
    # Verify the document in the database
    processed_doc = get_document_by_id(test_session, result.document_id)
    assert processed_doc is not None
    assert processed_doc.doc_type == "payment_app"
    assert processed_doc.party == "contractor"
    assert len(processed_doc.line_items) > 0


def test_process_multiple_documents(test_session, sample_payment_app, sample_change_order):
    """Test processing multiple documents and verifying their relationships."""
    payment_app, payment_app_path = sample_payment_app
    change_order, change_order_path = sample_change_order
    
    # Process the documents
    factory = DocumentProcessorFactory()
    processor = factory.create_processor(test_session)
    
    # Process payment app
    payment_result = processor.process_document(
        payment_app_path,
        "payment_app",
        "contractor",
        save_to_db=True,
        project_id="test_project"
    )
    
    # Process change order
    change_order_result = processor.process_document(
        change_order_path,
        "change_order",
        "contractor",
        save_to_db=True,
        project_id="test_project"
    )
    
    # Verify the results
    assert payment_result.success
    assert change_order_result.success
    
    # Verify documents in the database
    docs = get_documents_by_type(test_session, "payment_app")
    assert len(docs) >= 1
    
    docs = get_documents_by_type(test_session, "change_order")
    assert len(docs) >= 1
    
    # Verify the processor result metadata contains the project_id
    assert payment_result.metadata.get("project_id") == "test_project"
    assert change_order_result.metadata.get("project_id") == "test_project"


def test_document_with_related_line_items(test_session, sample_payment_app, sample_change_order):
    """Test that related line items can be identified across documents."""
    payment_app, _ = sample_payment_app
    change_order, _ = sample_change_order
    
    # Query the database for electrical line items
    electrical_items = test_session.query(LineItem).filter(
        LineItem.category == "electrical"
    ).all()
    
    # Verify we have electrical items from both documents
    assert len(electrical_items) >= 2
    
    # Find the corresponding document for each electrical item
    docs = set()
    for item in electrical_items:
        doc = test_session.query(Document).filter(
            Document.doc_id == item.doc_id
        ).one()
        docs.add(doc.doc_type)
    
    # Verify we have items from different document types
    assert "payment_app" in docs
    assert "change_order" in docs