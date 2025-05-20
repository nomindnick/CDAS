"""
Pytest fixtures for integration tests.

These fixtures set up a complete test environment for integration testing,
including database, sample documents, and mock file system.
"""

import os
import shutil
import tempfile
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cdas.db.models import Base, Document, LineItem, Page, Annotation
from cdas.db.operations import generate_document_id
from cdas.document_processor.factory import DocumentProcessorFactory
from cdas.financial_analysis.engine import FinancialAnalysisEngine
from cdas.config import set_config


@pytest.fixture(scope="session")
def test_db_path():
    """Create a temporary database file."""
    handle, path = tempfile.mkstemp(suffix='.db')
    yield path
    os.close(handle)
    os.unlink(path)


@pytest.fixture(scope="session")
def test_docs_dir():
    """Create a temporary directory for test documents."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def test_config(test_db_path):
    """Create a test configuration."""
    config = {
        "database": {
            "db_type": "sqlite",
            "db_path": test_db_path,
            "echo": False
        },
        "logging": {
            "level": "ERROR",
            "file": None
        }
    }
    set_config(config)
    return config


@pytest.fixture(scope="session")
def test_engine(test_config):
    """Create a test database engine."""
    engine = create_engine(f"sqlite:///{test_config['database']['db_path']}")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def test_session(test_engine):
    """Create a test database session."""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    
    yield session
    
    # Roll back any uncommitted changes after each test
    session.rollback()
    
    # Clean up the session
    session.close()


@pytest.fixture(scope="function")
def document_processor(test_session):
    """Create a document processor for testing."""
    factory = DocumentProcessorFactory()
    processor = factory.create_processor(test_session)
    return processor


@pytest.fixture(scope="function")
def analysis_engine(test_session):
    """Create a financial analysis engine for testing."""
    engine = FinancialAnalysisEngine(test_session)
    return engine


@pytest.fixture(scope="function")
def sample_payment_app(test_session, test_docs_dir, request):
    """Create a sample payment application document for testing."""
    # Create a simple payment application file
    file_content = """
    PAYMENT APPLICATION
    
    Project: Test Project
    Contractor: ABC Construction
    Date: 2023-01-15
    
    Item    Description                 Amount
    ------------------------------------------
    1       Foundation Work             $50,000.00
    2       Framing                     $75,000.00
    3       Electrical                  $35,000.00
    4       Plumbing                    $40,000.00
    
    Total Amount Due: $200,000.00
    """
    
    # Add a unique identifier to prevent file_hash collisions across tests
    test_name = request.node.name if hasattr(request, 'node') else 'unknown'
    unique_filename = f"payment_app_sample_{test_name}_{os.getpid()}.txt"
    file_path = os.path.join(test_docs_dir, unique_filename)
    with open(file_path, "w") as f:
        f.write(file_content)
    
    # Create database record with a unique hash
    unique_hash = f"test_hash_payment_app_{test_name}_{os.getpid()}"
    doc = Document(
        doc_id=generate_document_id(),
        file_name=unique_filename,
        file_path=file_path,
        file_hash=unique_hash,
        file_size=len(file_content),
        file_type="txt",
        doc_type="payment_app",
        party="contractor",
        status="active",
        meta_data={"project_id": "test_project"}
    )
    
    # Add pages
    page = Page(
        page_number=1,
        content=file_content,
        meta_data={}
    )
    doc.pages.append(page)
    
    # Add line items with unique IDs based on test name
    line_items = [
        LineItem(
            item_id=f"item_1_{test_name}_{os.getpid()}",
            item_number="1",
            description="Foundation Work",
            amount=50000.00,
            category="construction",
            status="pending",
            meta_data={}
        ),
        LineItem(
            item_id=f"item_2_{test_name}_{os.getpid()}",
            item_number="2",
            description="Framing",
            amount=75000.00,
            category="construction",
            status="pending",
            meta_data={}
        ),
        LineItem(
            item_id=f"item_3_{test_name}_{os.getpid()}",
            item_number="3",
            description="Electrical",
            amount=35000.00,
            category="electrical",
            status="pending",
            meta_data={}
        ),
        LineItem(
            item_id=f"item_4_{test_name}_{os.getpid()}",
            item_number="4",
            description="Plumbing",
            amount=40000.00,
            category="plumbing",
            status="pending",
            meta_data={}
        )
    ]
    
    for item in line_items:
        doc.line_items.append(item)
    
    test_session.add(doc)
    test_session.commit()
    
    return doc, file_path


@pytest.fixture(scope="function")
def sample_change_order(test_session, test_docs_dir, request):
    """Create a sample change order document for testing."""
    # Create a simple change order file
    file_content = """
    CHANGE ORDER
    
    Project: Test Project
    Contractor: ABC Construction
    Date: 2023-02-10
    
    Item    Description                         Amount
    --------------------------------------------------
    1       Additional Electrical Work           $15,000.00
    2       Extended Foundation Requirements     $25,000.00
    
    Total Change Amount: $40,000.00
    """
    
    # Add a unique identifier to prevent file_hash collisions across tests
    test_name = request.node.name if hasattr(request, 'node') else 'unknown'
    unique_filename = f"change_order_sample_{test_name}_{os.getpid()}.txt"
    file_path = os.path.join(test_docs_dir, unique_filename)
    with open(file_path, "w") as f:
        f.write(file_content)
    
    # Create database record with a unique hash
    unique_hash = f"test_hash_change_order_{test_name}_{os.getpid()}"
    doc = Document(
        doc_id=generate_document_id(),
        file_name=unique_filename,
        file_path=file_path,
        file_hash=unique_hash,
        file_size=len(file_content),
        file_type="txt",
        doc_type="change_order",
        party="contractor",
        status="active",
        meta_data={"project_id": "test_project"}
    )
    
    # Add pages
    page = Page(
        page_number=1,
        content=file_content,
        meta_data={}
    )
    doc.pages.append(page)
    
    # Add line items with unique IDs based on test name
    line_items = [
        LineItem(
            item_id=f"item_co_1_{test_name}_{os.getpid()}",
            item_number="1",
            description="Additional Electrical Work",
            amount=15000.00,
            category="electrical",
            status="pending",
            meta_data={}
        ),
        LineItem(
            item_id=f"item_co_2_{test_name}_{os.getpid()}",
            item_number="2",
            description="Extended Foundation Requirements",
            amount=25000.00,
            category="construction",
            status="pending",
            meta_data={}
        )
    ]
    
    for item in line_items:
        doc.line_items.append(item)
    
    test_session.add(doc)
    test_session.commit()
    
    return doc, file_path