"""
Database session management for the Construction Document Analysis System.

This module provides functions for creating and managing database sessions,
including creating the database engine and session factory.
"""

import os
from typing import Optional, Dict, Any, Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine

from cdas.config import get_config


def get_database_engine(config: Optional[Dict[str, Any]] = None) -> Engine:
    """Get SQLAlchemy engine based on configuration.
    
    Args:
        config: Configuration dictionary (defaults to global config if None)
        
    Returns:
        SQLAlchemy engine
    """
    if config is None:
        config = get_config().get('database', {})
    
    db_type = config.get('db_type', 'sqlite')
    
    if db_type == 'sqlite':
        db_path = config.get('db_path', 'cdas.db')
        # Make sure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        connection_string = f"sqlite:///{db_path}"
    elif db_type == 'postgresql':
        host = config.get('db_host', 'localhost')
        port = config.get('db_port', 5432)
        database = config.get('db_name', 'cdas')
        user = config.get('db_user', 'postgres')
        password = config.get('db_password', '')
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    
    # Configure engine
    connect_args = {}
    if db_type == 'sqlite':
        connect_args['check_same_thread'] = False
    
    engine = create_engine(
        connection_string,
        echo=config.get('echo', False),
        future=True,
        connect_args=connect_args
    )
    
    return engine


# Global engine and session factory
_engine = None
_SessionFactory = None


def init_db(config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize the database engine and session factory.
    
    Args:
        config: Configuration dictionary (defaults to global config if None)
    """
    global _engine, _SessionFactory
    _engine = get_database_engine(config)
    _SessionFactory = sessionmaker(bind=_engine)


def get_engine() -> Engine:
    """Get the database engine.
    
    Returns:
        SQLAlchemy engine
    """
    global _engine
    if _engine is None:
        init_db()
    return _engine


def get_session_factory() -> sessionmaker:
    """Get the session factory.
    
    Returns:
        SQLAlchemy sessionmaker
    """
    global _SessionFactory
    if _SessionFactory is None:
        init_db()
    return _SessionFactory


def get_session() -> Session:
    """Get a new database session.
    
    Returns:
        SQLAlchemy session
    """
    return get_session_factory()()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations.
    
    Yields:
        SQLAlchemy session
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()