"""
Pytest fixtures for database tests.
"""

import os
import tempfile
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from cdas.db.models import Base
from cdas.config import set_config


@pytest.fixture(scope="session")
def test_db_path():
    """Create a temporary database file."""
    handle, path = tempfile.mkstemp(suffix='.db')
    yield path
    os.close(handle)
    os.unlink(path)


@pytest.fixture(scope="session")
def test_config(test_db_path):
    """Create a test configuration."""
    config = {
        "database": {
            "db_type": "sqlite",
            "db_path": test_db_path,
            "echo": False
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
def sample_document(test_session):
    """Create a sample document for testing."""
    from cdas.db.models import Document
    from cdas.db.operations import generate_document_id
    import uuid
    
    # Create a unique hash for each test
    unique_hash = f"test_hash_{uuid.uuid4().hex}"
    
    doc = Document(
        doc_id=generate_document_id(),
        file_name="test_document.pdf",
        file_path="/path/to/test_document.pdf",
        file_hash=unique_hash,
        file_size=1024,
        file_type="pdf",
        doc_type="change_order",
        party="contractor",
        status="active",
        meta_data={"test_key": "test_value"}
    )
    
    test_session.add(doc)
    test_session.commit()
    
    return doc