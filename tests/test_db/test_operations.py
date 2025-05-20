"""
Tests for database operations.
"""

import os
import pytest
from datetime import datetime, date
from decimal import Decimal
import tempfile
import hashlib

from cdas.db.operations import (
    calculate_file_hash, generate_document_id, register_document,
    register_page, store_line_items, store_annotations,
    create_document_relationship, find_matching_amounts,
    create_amount_match, find_related_documents, create_analysis_flag,
    register_change_order, register_payment_application, create_report,
    add_report_evidence, create_financial_transaction, search_documents,
    search_line_items, get_change_orders_by_status,
    get_payment_applications_by_period, get_analysis_flags_by_type,
    delete_document
)

from cdas.db.models import (
    Document, Page, LineItem, DocumentRelationship, FinancialTransaction,
    Annotation, AnalysisFlag, Report, ReportEvidence, AmountMatch,
    ChangeOrder, PaymentApplication
)


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, 'w') as f:
        f.write("Test file content for hashing")
    yield path
    os.unlink(path)


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_calculate_file_hash(self, temp_file):
        """Test calculating file hash."""
        # Calculate hash using the function
        hash_result = calculate_file_hash(temp_file)
        
        # Calculate hash manually for comparison
        manual_hash = hashlib.sha256()
        with open(temp_file, 'rb') as f:
            manual_hash.update(f.read())
        expected_hash = manual_hash.hexdigest()
        
        assert hash_result == expected_hash
    
    def test_generate_document_id(self):
        """Test generating document ID."""
        # Generate a document ID
        doc_id = generate_document_id()
        
        # Check that it's a string of the right format (UUID)
        assert isinstance(doc_id, str)
        assert len(doc_id) == 36  # UUID format length
        
        # Check that multiple calls generate different IDs
        another_doc_id = generate_document_id()
        assert doc_id != another_doc_id


class TestDocumentOperations:
    """Tests for document operations."""
    
    def test_register_document(self, test_session, temp_file):
        """Test registering a document."""
        # Register a document
        doc = register_document(
            test_session,
            temp_file,
            doc_type="contract",
            party="district",
            date_created=date(2023, 1, 15),
            date_received=date(2023, 1, 20),
            processed_by="test_user",
            metadata={"project": "Test Project"}
        )
        
        # Check that document was created
        assert doc is not None
        assert isinstance(doc, Document)
        assert doc.file_path == temp_file
        assert doc.doc_type == "contract"
        assert doc.party == "district"
        assert doc.date_created == date(2023, 1, 15)
        assert doc.date_received == date(2023, 1, 20)
        assert doc.processed_by == "test_user"
        assert doc.meta_data == {"project": "Test Project"}
        
        # Check that document is in the database
        queried_doc = test_session.query(Document).filter_by(doc_id=doc.doc_id).first()
        assert queried_doc is not None
        assert queried_doc.doc_id == doc.doc_id
    
    def test_register_existing_document(self, test_session, temp_file):
        """Test registering a document that already exists."""
        # Register a document
        doc1 = register_document(
            test_session,
            temp_file,
            doc_type="contract",
            party="district"
        )
        
        # Register the same document again
        doc2 = register_document(
            test_session,
            temp_file,
            doc_type="different_type",  # Different type, should be ignored
            party="different_party"     # Different party, should be ignored
        )
        
        # Check that the same document was returned
        assert doc1 is not None
        assert doc2 is not None
        assert doc1.doc_id == doc2.doc_id
        assert doc2.doc_type == "contract"  # Original type preserved
        assert doc2.party == "district"     # Original party preserved


class TestPageOperations:
    """Tests for page operations."""
    
    def test_register_page(self, test_session, sample_document):
        """Test registering a page."""
        # Register a page
        page = register_page(
            test_session,
            sample_document.doc_id,
            page_number=1,
            content="This is a test page",
            has_tables=True,
            has_handwriting=False,
            has_financial_data=True
        )
        
        # Check that page was created
        assert page is not None
        assert isinstance(page, Page)
        assert page.doc_id == sample_document.doc_id
        assert page.page_number == 1
        assert page.content == "This is a test page"
        assert page.has_tables is True
        assert page.has_handwriting is False
        assert page.has_financial_data is True
        
        # Check that page is in the database
        queried_page = test_session.query(Page).filter_by(doc_id=sample_document.doc_id, page_number=1).first()
        assert queried_page is not None
        assert queried_page.page_id == page.page_id
    
    def test_register_existing_page(self, test_session, sample_document):
        """Test registering a page that already exists."""
        # Register a page
        page1 = register_page(
            test_session,
            sample_document.doc_id,
            page_number=2,
            content="Original content",
            has_tables=False
        )
        
        # Register the same page again with different content
        page2 = register_page(
            test_session,
            sample_document.doc_id,
            page_number=2,
            content="Updated content",
            has_tables=True
        )
        
        # Check that the page was updated
        assert page1 is not None
        assert page2 is not None
        assert page1.page_id == page2.page_id
        assert page2.content == "Updated content"
        assert page2.has_tables is True
        
        # Check that the page in the database was updated
        queried_page = test_session.query(Page).filter_by(doc_id=sample_document.doc_id, page_number=2).first()
        assert queried_page is not None
        assert queried_page.content == "Updated content"


class TestLineItemOperations:
    """Tests for line item operations."""
    
    def test_store_line_items(self, test_session, sample_document):
        """Test storing line items."""
        # Create a page
        page = register_page(
            test_session,
            sample_document.doc_id,
            page_number=3
        )
        
        # Store line items
        line_items_data = [
            {
                "page_id": page.page_id,
                "description": "Item 1",
                "amount": 100.50,
                "quantity": 2,
                "unit_price": 50.25,
                "total": 100.50,
                "cost_code": "123-456",
                "category": "labor",
                "status": "approved",
                "location_in_doc": {"x": 100, "y": 200},
                "confidence": 0.95,
                "context": "From table on page 3",
                "metadata": {"source": "table_extraction"}
            },
            {
                "page_id": page.page_id,
                "description": "Item 2",
                "amount": 200.75,
                "quantity": 1,
                "unit_price": 200.75,
                "total": 200.75,
                "cost_code": "123-789",
                "category": "materials",
                "status": "pending",
                "location_in_doc": {"x": 100, "y": 250},
                "confidence": 0.90,
                "context": "From table on page 3",
                "metadata": {"source": "table_extraction"}
            }
        ]
        
        result = store_line_items(
            test_session,
            sample_document.doc_id,
            line_items_data
        )
        
        # Check that line items were created
        assert result is not None
        assert len(result) == 2
        assert all(isinstance(item, LineItem) for item in result)
        
        # Check that line items are in the database
        queried_items = test_session.query(LineItem).filter_by(doc_id=sample_document.doc_id).all()
        assert len(queried_items) == 2
        
        # Check that line items have correct data
        item1 = next((item for item in queried_items if item.description == "Item 1"), None)
        item2 = next((item for item in queried_items if item.description == "Item 2"), None)
        
        assert item1 is not None
        assert float(item1.amount) == 100.50
        assert item1.cost_code == "123-456"
        
        assert item2 is not None
        assert float(item2.amount) == 200.75
        assert item2.category == "materials"


class TestDocumentRelationshipOperations:
    """Tests for document relationship operations."""
    
    def test_create_document_relationship(self, test_session):
        """Test creating a document relationship."""
        # Create two documents
        doc1 = register_document(
            test_session,
            "/path/to/doc1.pdf",
            doc_type="contract"
        )
        
        doc2 = register_document(
            test_session,
            "/path/to/doc2.pdf",
            doc_type="change_order"
        )
        
        # Create a relationship
        rel = create_document_relationship(
            test_session,
            doc1.doc_id,
            doc2.doc_id,
            "references",
            confidence=0.9,
            metadata={"notes": "Contract references change order"}
        )
        
        # Check that relationship was created
        assert rel is not None
        assert isinstance(rel, DocumentRelationship)
        assert rel.source_doc_id == doc1.doc_id
        assert rel.target_doc_id == doc2.doc_id
        assert rel.relationship_type == "references"
        assert float(rel.confidence) == 0.9
        assert rel.metadata == {"notes": "Contract references change order"}
        
        # Check that relationship is in the database
        queried_rel = test_session.query(DocumentRelationship).filter_by(
            source_doc_id=doc1.doc_id,
            target_doc_id=doc2.doc_id
        ).first()
        
        assert queried_rel is not None
        assert queried_rel.relationship_id == rel.relationship_id
    
    def test_create_existing_document_relationship(self, test_session):
        """Test creating a document relationship that already exists."""
        # Create two documents
        doc1 = register_document(
            test_session,
            "/path/to/doc3.pdf",
            doc_type="contract"
        )
        
        doc2 = register_document(
            test_session,
            "/path/to/doc4.pdf",
            doc_type="change_order"
        )
        
        # Create a relationship
        rel1 = create_document_relationship(
            test_session,
            doc1.doc_id,
            doc2.doc_id,
            "references",
            confidence=0.8
        )
        
        # Create the same relationship again with different confidence
        rel2 = create_document_relationship(
            test_session,
            doc1.doc_id,
            doc2.doc_id,
            "references",
            confidence=0.9
        )
        
        # Check that the relationship was updated
        assert rel1 is not None
        assert rel2 is not None
        assert rel1.relationship_id == rel2.relationship_id
        assert float(rel2.confidence) == 0.9
        
        # Check that only one relationship exists in the database
        relationships = test_session.query(DocumentRelationship).filter_by(
            source_doc_id=doc1.doc_id,
            target_doc_id=doc2.doc_id,
            relationship_type="references"
        ).all()
        
        assert len(relationships) == 1


class TestAmountMatchOperations:
    """Tests for amount match operations."""
    
    def test_find_matching_amounts(self, test_session, sample_document):
        """Test finding matching amounts."""
        # Create pages
        page1 = register_page(
            test_session,
            sample_document.doc_id,
            page_number=4
        )
        
        # Create line items with different amounts
        line_items_data = [
            {
                "description": "Exact Match Item",
                "amount": 1000.00
            },
            {
                "description": "Close Match Item 1",
                "amount": 999.99
            },
            {
                "description": "Close Match Item 2",
                "amount": 1000.01
            },
            {
                "description": "Non-Match Item",
                "amount": 2000.00
            }
        ]
        
        line_items = store_line_items(
            test_session,
            sample_document.doc_id,
            line_items_data
        )
        
        # Find matching amounts
        matches = find_matching_amounts(
            test_session,
            1000.00,
            tolerance=0.02
        )
        
        # Check that matching items were found
        assert matches is not None
        assert len(matches) == 3  # Exact match + 2 close matches
        
        # Check that the right items were matched
        matched_descriptions = [item.description for item in matches]
        assert "Exact Match Item" in matched_descriptions
        assert "Close Match Item 1" in matched_descriptions
        assert "Close Match Item 2" in matched_descriptions
        assert "Non-Match Item" not in matched_descriptions
    
    def test_create_amount_match(self, test_session, sample_document):
        """Test creating an amount match."""
        # Create line items
        line_items_data = [
            {"description": "Source Item", "amount": 1500.00},
            {"description": "Target Item", "amount": 1500.00}
        ]
        
        line_items = store_line_items(
            test_session,
            sample_document.doc_id,
            line_items_data
        )
        
        source_item = next((item for item in line_items if item.description == "Source Item"), None)
        target_item = next((item for item in line_items if item.description == "Target Item"), None)
        
        # Create an amount match
        match = create_amount_match(
            test_session,
            source_item.item_id,
            target_item.item_id,
            match_type="exact",
            confidence=1.0,
            difference=0.0,
            metadata={"note": "Exact match"}
        )
        
        # Check that match was created
        assert match is not None
        assert isinstance(match, AmountMatch)
        assert match.source_item_id == source_item.item_id
        assert match.target_item_id == target_item.item_id
        assert match.match_type == "exact"
        assert float(match.confidence) == 1.0
        assert float(match.difference) == 0.0
        assert match.metadata == {"note": "Exact match"}
        
        # Check that match is in the database
        queried_match = test_session.query(AmountMatch).filter_by(
            source_item_id=source_item.item_id,
            target_item_id=target_item.item_id
        ).first()
        
        assert queried_match is not None
        assert queried_match.match_id == match.match_id