"""
Pytest fixtures for testing the reporting system.
"""

import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cdas.db.models import Base
from cdas.reporting.generator import ReportGenerator


@pytest.fixture
def engine():
    """Create a SQLAlchemy engine for testing."""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    """Create a SQLAlchemy session for testing."""
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def report_generator(session):
    """Create a report generator for testing."""
    return ReportGenerator(session)