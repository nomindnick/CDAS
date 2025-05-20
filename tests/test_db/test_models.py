"""
Tests for database models.
"""

import pytest
from sqlalchemy.exc import IntegrityError

from cdas.db.models import (
    Document, Page, LineItem, DocumentRelationship, FinancialTransaction,
    Annotation, AnalysisFlag, Report, ReportEvidence, AmountMatch,
    ChangeOrder, PaymentApplication
)


class TestDocumentModel:
    """Tests for the Document model."""
    
    def test_document_creation(self, test_session):
        """Test creating a document."""
        # Create a document
        doc = Document(
            doc_id="test-doc-001",
            file_name="test.pdf",
            file_path="/path/to/test.pdf",
            file_hash="abc123",
            file_size=1024,
            file_type="pdf",
            doc_type="change_order",
            party="contractor",
            status="active",
            meta_data={"key": "value"}
        )
        
        # Add to session and commit
        test_session.add(doc)
        test_session.commit()
        
        # Query the document
        queried_doc = test_session.query(Document).filter_by(doc_id="test-doc-001").first()
        
        # Check attributes
        assert queried_doc is not None
        assert queried_doc.doc_id == "test-doc-001"
        assert queried_doc.file_name == "test.pdf"
        assert queried_doc.file_path == "/path/to/test.pdf"
        assert queried_doc.file_hash == "abc123"
        assert queried_doc.file_size == 1024
        assert queried_doc.file_type == "pdf"
        assert queried_doc.doc_type == "change_order"
        assert queried_doc.party == "contractor"
        assert queried_doc.status == "active"
        assert queried_doc.meta_data == {"key": "value"}
    
    def test_document_unique_constraints(self, test_session):
        """Test unique constraints on the Document model."""
        # Create a document
        doc1 = Document(
            doc_id="test-doc-002",
            file_name="test2.pdf",
            file_path="/path/to/test2.pdf",
            file_hash="def456",
            file_size=2048,
            file_type="pdf"
        )
        
        test_session.add(doc1)
        test_session.commit()
        
        # Create another document with the same file hash
        doc2 = Document(
            doc_id="test-doc-003",
            file_name="test3.pdf",
            file_path="/path/to/test3.pdf",
            file_hash="def456",  # Same as doc1
            file_size=3072,
            file_type="pdf"
        )
        
        test_session.add(doc2)
        
        # Should raise IntegrityError due to unique constraint on file_hash
        with pytest.raises(IntegrityError):
            test_session.commit()
        
        # Rollback the session
        test_session.rollback()


class TestPageModel:
    """Tests for the Page model."""
    
    def test_page_creation(self, test_session, sample_document):
        """Test creating a page."""
        # Create a page
        page = Page(
            doc_id=sample_document.doc_id,
            page_number=1,
            content="This is page 1",
            has_tables=True,
            has_handwriting=False,
            has_financial_data=True
        )
        
        # Add to session and commit
        test_session.add(page)
        test_session.commit()
        
        # Query the page
        queried_page = test_session.query(Page).filter_by(doc_id=sample_document.doc_id, page_number=1).first()
        
        # Check attributes
        assert queried_page is not None
        assert queried_page.doc_id == sample_document.doc_id
        assert queried_page.page_number == 1
        assert queried_page.content == "This is page 1"
        assert queried_page.has_tables is True
        assert queried_page.has_handwriting is False
        assert queried_page.has_financial_data is True
        
        # Check relationship with document
        assert queried_page.document is not None
        assert queried_page.document.doc_id == sample_document.doc_id
    
    def test_page_unique_constraints(self, test_session, sample_document):
        """Test unique constraints on the Page model."""
        # Create a page
        page1 = Page(
            doc_id=sample_document.doc_id,
            page_number=2,
            content="This is page 2"
        )
        
        test_session.add(page1)
        test_session.commit()
        
        # Create another page with the same document and page number
        page2 = Page(
            doc_id=sample_document.doc_id,
            page_number=2,  # Same as page1
            content="This is also page 2"
        )
        
        test_session.add(page2)
        
        # Should raise IntegrityError due to unique constraint on (doc_id, page_number)
        with pytest.raises(IntegrityError):
            test_session.commit()
        
        # Rollback the session
        test_session.rollback()


class TestLineItemModel:
    """Tests for the LineItem model."""
    
    def test_line_item_creation(self, test_session, sample_document):
        """Test creating a line item."""
        # Create a page
        page = Page(
            doc_id=sample_document.doc_id,
            page_number=3,
            content="This is page 3"
        )
        
        test_session.add(page)
        test_session.commit()
        
        # Create a line item
        line_item = LineItem(
            doc_id=sample_document.doc_id,
            page_id=page.page_id,
            description="Test line item",
            amount=1000.50,
            quantity=2.0,
            unit_price=500.25,
            total=1000.50,
            cost_code="123-456",
            category="labor",
            status="approved",
            location_in_doc={"x": 100, "y": 200, "width": 300, "height": 50},
            extraction_confidence=0.95,
            context="Item appears in change order table",
            meta_data={"source": "table_extraction"}
        )
        
        # Add to session and commit
        test_session.add(line_item)
        test_session.commit()
        
        # Query the line item
        queried_item = test_session.query(LineItem).filter_by(doc_id=sample_document.doc_id, description="Test line item").first()
        
        # Check attributes
        assert queried_item is not None
        assert queried_item.doc_id == sample_document.doc_id
        assert queried_item.page_id == page.page_id
        assert queried_item.description == "Test line item"
        assert float(queried_item.amount) == 1000.50
        assert float(queried_item.quantity) == 2.0
        assert float(queried_item.unit_price) == 500.25
        assert float(queried_item.total) == 1000.50
        assert queried_item.cost_code == "123-456"
        assert queried_item.category == "labor"
        assert queried_item.status == "approved"
        assert queried_item.location_in_doc == {"x": 100, "y": 200, "width": 300, "height": 50}
        assert float(queried_item.extraction_confidence) == 0.95
        assert queried_item.context == "Item appears in change order table"
        assert queried_item.meta_data == {"source": "table_extraction"}
        
        # Check relationships
        assert queried_item.document is not None
        assert queried_item.document.doc_id == sample_document.doc_id
        assert queried_item.page is not None
        assert queried_item.page.page_id == page.page_id


class TestDocumentRelationshipModel:
    """Tests for the DocumentRelationship model."""
    
    def test_document_relationship_creation(self, test_session):
        """Test creating a document relationship."""
        # Create two documents
        doc1 = Document(
            doc_id="test-doc-rel-001",
            file_name="source.pdf",
            file_path="/path/to/source.pdf",
            file_hash="rel-hash-001",
            file_size=1024,
            file_type="pdf"
        )
        
        doc2 = Document(
            doc_id="test-doc-rel-002",
            file_name="target.pdf",
            file_path="/path/to/target.pdf",
            file_hash="rel-hash-002",
            file_size=2048,
            file_type="pdf"
        )
        
        test_session.add_all([doc1, doc2])
        test_session.commit()
        
        # Create a relationship
        rel = DocumentRelationship(
            source_doc_id=doc1.doc_id,
            target_doc_id=doc2.doc_id,
            relationship_type="references",
            confidence=0.9,
            meta_data={"notes": "Document 1 references Document 2"}
        )
        
        # Add to session and commit
        test_session.add(rel)
        test_session.commit()
        
        # Query the relationship
        queried_rel = test_session.query(DocumentRelationship).filter_by(
            source_doc_id=doc1.doc_id,
            target_doc_id=doc2.doc_id
        ).first()
        
        # Check attributes
        assert queried_rel is not None
        assert queried_rel.source_doc_id == doc1.doc_id
        assert queried_rel.target_doc_id == doc2.doc_id
        assert queried_rel.relationship_type == "references"
        assert float(queried_rel.confidence) == 0.9
        assert queried_rel.meta_data == {"notes": "Document 1 references Document 2"}
        
        # Check relationships
        assert queried_rel.source_document is not None
        assert queried_rel.source_document.doc_id == doc1.doc_id
        assert queried_rel.target_document is not None
        assert queried_rel.target_document.doc_id == doc2.doc_id


class TestCascadeDelete:
    """Tests for cascade delete behavior."""
    
    def test_document_cascade_delete(self, test_session):
        """Test that deleting a document cascades to related entities."""
        # Create a document
        doc = Document(
            doc_id="test-cascade-001",
            file_name="cascade.pdf",
            file_path="/path/to/cascade.pdf",
            file_hash="cascade-hash-001",
            file_size=1024,
            file_type="pdf"
        )
        
        test_session.add(doc)
        test_session.commit()
        
        # Create a page
        page = Page(
            doc_id=doc.doc_id,
            page_number=1,
            content="Cascade page"
        )
        
        test_session.add(page)
        test_session.commit()
        
        # Create a line item
        line_item = LineItem(
            doc_id=doc.doc_id,
            page_id=page.page_id,
            description="Cascade line item",
            amount=100.0
        )
        
        test_session.add(line_item)
        test_session.commit()
        
        # Check that everything exists
        assert test_session.query(Document).filter_by(doc_id=doc.doc_id).count() == 1
        assert test_session.query(Page).filter_by(doc_id=doc.doc_id).count() == 1
        assert test_session.query(LineItem).filter_by(doc_id=doc.doc_id).count() == 1
        
        # Delete the document
        test_session.delete(doc)
        test_session.commit()
        
        # Check that everything is deleted
        assert test_session.query(Document).filter_by(doc_id=doc.doc_id).count() == 0
        assert test_session.query(Page).filter_by(doc_id=doc.doc_id).count() == 0
        assert test_session.query(LineItem).filter_by(doc_id=doc.doc_id).count() == 0