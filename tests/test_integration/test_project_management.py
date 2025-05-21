"""
Integration tests for project management functionality.

These tests verify that the project database manager works correctly
by creating, using, and deleting project databases.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from cdas.db.project_manager import get_project_db_manager, get_session, session_scope
from cdas.db.models import Document, LineItem, Page
from cdas.document_processor.factory import DocumentProcessorFactory
from cdas.document_processor.processor import DocumentType, PartyType
from cdas.utils.project_ops import run_across_projects, find_in_all_projects


class TestProjectManagement:
    """Test project database management functionality."""
    
    @pytest.fixture(scope="class")
    def temp_db_dir(self):
        """Create a temporary directory for test databases."""
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="cdas_test_")
        yield temp_dir
        # Clean up after tests
        shutil.rmtree(temp_dir)
    
    @pytest.fixture(scope="class")
    def project_manager(self, temp_db_dir):
        """Set up project manager with a temporary database directory."""
        # Get project manager and configure it to use temp directory
        manager = get_project_db_manager()
        original_db_root = manager.db_root
        manager.db_root = Path(temp_db_dir)
        
        yield manager
        
        # Reset to original directory
        manager.db_root = original_db_root
    
    @pytest.fixture(scope="class")
    def test_projects(self, project_manager):
        """Create test projects for testing."""
        # Create test projects
        test_project_ids = ["test_project_1", "test_project_2"]
        for project_id in test_project_ids:
            project_manager.create_project(project_id)
        
        yield test_project_ids
        
        # Clean up test projects
        for project_id in test_project_ids:
            if project_id in project_manager.get_project_list():
                project_manager.delete_project(project_id)
    
    def test_project_creation(self, project_manager):
        """Test creating a new project."""
        # Create a new project
        project_id = "creation_test"
        assert project_manager.create_project(project_id)
        
        # Verify it appears in the list
        assert project_id in project_manager.get_project_list()
        
        # Clean up
        project_manager.delete_project(project_id)
    
    def test_project_deletion(self, project_manager):
        """Test deleting a project."""
        # Create a project to delete
        project_id = "deletion_test"
        project_manager.create_project(project_id)
        
        # Verify it was created
        assert project_id in project_manager.get_project_list()
        
        # Delete it
        assert project_manager.delete_project(project_id)
        
        # Verify it's gone
        assert project_id not in project_manager.get_project_list()
    
    def test_invalid_project_id(self, project_manager):
        """Test handling of invalid project IDs."""
        # Test invalid project names
        assert not project_manager.create_project("invalid/name")
        assert not project_manager.create_project("spaces not allowed")
        assert not project_manager.create_project("")
    
    def test_project_context_switching(self, project_manager, test_projects):
        """Test switching between projects."""
        # Set first project as current
        project_manager.set_current_project(test_projects[0])
        assert project_manager.get_current_project() == test_projects[0]
        
        # Switch to second project
        project_manager.set_current_project(test_projects[1])
        assert project_manager.get_current_project() == test_projects[1]
    
    def test_project_data_isolation(self, project_manager, test_projects):
        """Test that projects are properly isolated."""
        # Add data to project 1
        with session_scope(test_projects[0]) as session:
            doc1 = Document(
                doc_id="doc1",
                doc_type=DocumentType.INVOICE.value,
                party=PartyType.CONTRACTOR.value,
                file_name="test_doc1.pdf"
            )
            session.add(doc1)
        
        # Add different data to project 2
        with session_scope(test_projects[1]) as session:
            doc2 = Document(
                doc_id="doc2",
                doc_type=DocumentType.CHANGE_ORDER.value,
                party=PartyType.OWNER.value,
                file_name="test_doc2.pdf"
            )
            session.add(doc2)
        
        # Verify project 1 has only its data
        with session_scope(test_projects[0]) as session:
            docs = session.query(Document).all()
            assert len(docs) == 1
            assert docs[0].doc_id == "doc1"
            assert docs[0].doc_type == DocumentType.INVOICE.value
            
            # Check project 2 docs aren't visible
            p2_docs = session.query(Document).filter(Document.doc_id == "doc2").all()
            assert len(p2_docs) == 0
            
        # Verify project 2 has only its data
        with session_scope(test_projects[1]) as session:
            docs = session.query(Document).all()
            assert len(docs) == 1
            assert docs[0].doc_id == "doc2"
            assert docs[0].doc_type == DocumentType.CHANGE_ORDER.value
            
            # Check project 1 docs aren't visible
            p1_docs = session.query(Document).filter(Document.doc_id == "doc1").all()
            assert len(p1_docs) == 0
    
    def test_document_processor_with_projects(self, project_manager, test_projects):
        """Test document processor with project databases."""
        factory = DocumentProcessorFactory()
        
        # Process document in project 1
        # This is a simplified test that mocks the document processing
        with session_scope(test_projects[0]) as session:
            # Mock document processing
            doc = Document(
                doc_id="test_doc",
                doc_type=DocumentType.INVOICE.value,
                party=PartyType.CONTRACTOR.value,
                file_name="test_doc.pdf"
            )
            session.add(doc)
            session.flush()
            
            # Add a line item
            item = LineItem(
                document_id=doc.id,
                amount=1000.00,
                description="Test item"
            )
            session.add(item)
        
        # Verify it's in project 1
        with session_scope(test_projects[0]) as session:
            items = session.query(LineItem).all()
            assert len(items) == 1
            assert items[0].amount == 1000.00
        
        # Verify it's NOT in project 2
        with session_scope(test_projects[1]) as session:
            items = session.query(LineItem).all()
            assert len(items) == 0
    
    def test_cross_project_operations(self, project_manager, test_projects):
        """Test operations across multiple projects."""
        # Helper function to count documents in a project
        def count_documents(session):
            return session.query(Document).count()
        
        # Add test data to both projects
        with session_scope(test_projects[0]) as session:
            session.add(Document(
                doc_id="p1_doc1",
                doc_type=DocumentType.INVOICE.value,
                party=PartyType.CONTRACTOR.value,
                file_name="p1_doc1.pdf"
            ))
            session.add(Document(
                doc_id="p1_doc2",
                doc_type=DocumentType.INVOICE.value,
                party=PartyType.CONTRACTOR.value,
                file_name="p1_doc2.pdf"
            ))
        
        with session_scope(test_projects[1]) as session:
            session.add(Document(
                doc_id="p2_doc1",
                doc_type=DocumentType.CHANGE_ORDER.value,
                party=PartyType.OWNER.value,
                file_name="p2_doc1.pdf"
            ))
        
        # Run count operation across both projects
        results = run_across_projects(count_documents, projects=test_projects)
        
        # Verify results
        assert len(results) == 2
        assert results[test_projects[0]] == 2  # Project 1 has 2 documents
        assert results[test_projects[1]] == 1  # Project 2 has 1 document
    
    def test_find_in_all_projects(self, project_manager, test_projects):
        """Test finding items across all projects."""
        # Helper function to find documents by type
        def find_by_type(session, doc_type):
            return session.query(Document).filter(Document.doc_type == doc_type).all()
        
        # Add test data if not already present
        with session_scope(test_projects[0]) as session:
            if session.query(Document).count() == 0:
                session.add(Document(
                    doc_id="p1_invoice",
                    doc_type=DocumentType.INVOICE.value,
                    party=PartyType.CONTRACTOR.value,
                    file_name="p1_invoice.pdf"
                ))
        
        with session_scope(test_projects[1]) as session:
            if session.query(Document).count() == 0:
                session.add(Document(
                    doc_id="p2_invoice",
                    doc_type=DocumentType.INVOICE.value,
                    party=PartyType.OWNER.value,
                    file_name="p2_invoice.pdf"
                ))
                session.add(Document(
                    doc_id="p2_change_order",
                    doc_type=DocumentType.CHANGE_ORDER.value,
                    party=PartyType.OWNER.value,
                    file_name="p2_change_order.pdf"
                ))
        
        # Find all invoices across projects
        results = find_in_all_projects(
            find_by_type,
            doc_type=DocumentType.INVOICE.value
        )
        
        # Verify results
        assert isinstance(results, list)
        assert len(results) >= 2  # At least the 2 invoices we added
        
        # Verify project info is included
        project_ids = {item['project_id'] for item in results}
        assert test_projects[0] in project_ids
        assert test_projects[1] in project_ids
        
        # Verify all results are invoices
        for item in results:
            assert 'doc_type' in item
            assert item['doc_type'] == DocumentType.INVOICE.value


class TestProjectBasedIngestion:
    """Test document ingestion with project database functionality."""
    
    @pytest.fixture(scope="class")
    def temp_db_dir(self):
        """Create a temporary directory for test databases."""
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="cdas_test_")
        yield temp_dir
        # Clean up after tests
        shutil.rmtree(temp_dir)
    
    @pytest.fixture(scope="class")
    def project_manager(self, temp_db_dir):
        """Set up project manager with a temporary database directory."""
        # Get project manager and configure it to use temp directory
        manager = get_project_db_manager()
        original_db_root = manager.db_root
        manager.db_root = Path(temp_db_dir)
        
        yield manager
        
        # Reset to original directory
        manager.db_root = original_db_root
    
    @pytest.fixture(scope="class")
    def ingestion_project(self, project_manager):
        """Create a test project for document ingestion tests."""
        project_id = "ingestion_test"
        project_manager.create_project(project_id)
        yield project_id
        project_manager.delete_project(project_id)
    
    @pytest.fixture(scope="function")
    def temp_text_file(self):
        """Create a temporary text file for document ingestion."""
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as file:
            file.write(b"Test document content\nLine 2\nAmount: $1234.56")
            temp_file_path = file.name
        
        yield temp_file_path
        
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
    
    def test_document_ingestion_with_project(self, project_manager, ingestion_project, temp_text_file):
        """Test document ingestion with project context."""
        factory = DocumentProcessorFactory()
        
        # Process document with project ID
        result = factory.process_single_document(
            file_path=temp_text_file,
            doc_type=DocumentType.CORRESPONDENCE,
            party=PartyType.CONTRACTOR,
            project_id=ingestion_project,
            save_to_db=True
        )
        
        assert result.success
        assert result.document_id is not None
        
        # Verify document was saved to the project database
        with session_scope(ingestion_project) as session:
            doc = session.query(Document).filter(Document.doc_id == result.document_id).first()
            assert doc is not None
            assert doc.doc_type == DocumentType.CORRESPONDENCE.value
            assert doc.party == PartyType.CONTRACTOR.value
            
            # Verify project ID is in metadata
            assert doc.metadata is not None
            assert 'project_id' in doc.metadata
            assert doc.metadata['project_id'] == ingestion_project
    
    def test_document_ingestion_with_session(self, project_manager, ingestion_project, temp_text_file):
        """Test document ingestion with explicit session."""
        factory = DocumentProcessorFactory()
        
        # Get session for project
        with session_scope(ingestion_project) as session:
            # Process document with explicit session
            result = factory.process_single_document(
                session=session,
                file_path=temp_text_file,
                doc_type=DocumentType.INVOICE,
                party=PartyType.OWNER,
                save_to_db=True
            )
            
            assert result.success
            assert result.document_id is not None
            
            # Verify document exists in database
            doc = session.query(Document).filter(Document.doc_id == result.document_id).first()
            assert doc is not None
            assert doc.doc_type == DocumentType.INVOICE.value
            assert doc.party == PartyType.OWNER.value