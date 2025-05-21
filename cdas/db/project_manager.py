"""
Project-based database management for the Construction Document Analysis System.

This module provides functionality for managing multiple project databases,
each in its own isolated database file or schema.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Optional, Union, List
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine

from cdas.config import get_config
from cdas.db.models import Base

logger = logging.getLogger(__name__)


class ProjectDatabaseManager:
    """Manages database connections for separate project databases."""
    
    def __init__(self):
        """Initialize the project database manager."""
        self.config = get_config()
        
        # Get database configuration
        db_config = self.config.get('database', {})
        
        # Get project database root directory (create if it doesn't exist)
        self.db_root = Path(self.config.get('project_database_root', 
                                            str(Path.home() / '.cdas' / 'projects')))
        os.makedirs(self.db_root, exist_ok=True)
        
        # Get database type
        self.db_type = db_config.get('db_type', 'sqlite')
        
        # Connection caching
        self.engines: Dict[str, Engine] = {}
        self.session_factories: Dict[str, sessionmaker] = {}
        
        # Current project context
        self.current_project: Optional[str] = None
    
    def get_project_list(self) -> List[str]:
        """Returns a list of available project databases.
        
        Returns:
            List of project IDs
        """
        if self.db_type == 'sqlite':
            return [p.stem for p in self.db_root.glob('*.sqlite')]
        else:
            # For PostgreSQL, would need to query available databases
            # This is a simplified implementation that just returns
            # the projects we've connected to in this session
            return list(self.engines.keys())
    
    def _get_connection_string(self, project_id: str) -> str:
        """Generate connection string for the specified project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Database connection string
        """
        db_config = self.config.get('database', {})
        
        if self.db_type == 'sqlite':
            db_path = self.db_root / f"{project_id}.sqlite"
            return f"sqlite:///{db_path}"
        else:
            # For PostgreSQL
            host = db_config.get('db_host', 'localhost')
            port = db_config.get('db_port', '5432')
            user = db_config.get('db_user', 'postgres')
            password = db_config.get('db_password', '')
            
            # Use either schema-based or database-based separation
            use_schemas = self.config.get('use_postgres_schemas', False)
            
            if use_schemas:
                # Use schema-based separation within the same database
                database = db_config.get('db_name', 'cdas')
                # Configure for schema usage
                return f"postgresql://{user}:{password}@{host}:{port}/{database}"
            else:
                # Use separate databases for each project
                # Project ID becomes the database name
                return f"postgresql://{user}:{password}@{host}:{port}/{project_id}"
    
    def create_project(self, project_id: str) -> bool:
        """Create a new project database.
        
        Args:
            project_id: Project identifier
            
        Returns:
            True if project was created successfully, False otherwise
        """
        if not self._is_valid_project_id(project_id):
            logger.error(f"Invalid project ID: {project_id}")
            return False
            
        if project_id in self.get_project_list():
            logger.warning(f"Project {project_id} already exists")
            return False
            
        try:
            connection_string = self._get_connection_string(project_id)
            engine = create_engine(connection_string)
            
            # Create tables
            Base.metadata.create_all(engine)
            
            # Store engine in cache
            self.engines[project_id] = engine
            self.session_factories[project_id] = sessionmaker(bind=engine)
            
            logger.info(f"Created database for project: {project_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create project {project_id}: {e}")
            return False
    
    def delete_project(self, project_id: str) -> bool:
        """Delete a project database.
        
        Args:
            project_id: Project identifier
            
        Returns:
            True if project was deleted successfully, False otherwise
        """
        if project_id not in self.get_project_list():
            logger.error(f"Project {project_id} does not exist")
            return False
        
        # Close any existing connections
        if project_id in self.engines:
            self.engines[project_id].dispose()
            del self.engines[project_id]
            
        if project_id in self.session_factories:
            del self.session_factories[project_id]
        
        # Reset current project if it's the one being deleted
        if self.current_project == project_id:
            self.current_project = None
        
        if self.db_type == 'sqlite':
            db_path = self.db_root / f"{project_id}.sqlite"
            try:
                if db_path.exists():
                    os.remove(db_path)
                logger.info(f"Deleted database for project: {project_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete project {project_id}: {e}")
                return False
        else:
            # For PostgreSQL, would need to drop database or schema
            # This requires admin privileges and careful implementation
            logger.error("PostgreSQL project deletion not implemented directly. Please use database administration tools.")
            return False
    
    def _is_valid_project_id(self, project_id: str) -> bool:
        """Check if a project ID is valid (alphanumeric plus underscores).
        
        Args:
            project_id: Project identifier to validate
            
        Returns:
            True if the project ID is valid, False otherwise
        """
        return bool(re.match(r'^[a-zA-Z0-9_]+$', project_id))
    
    def get_engine(self, project_id: str) -> Engine:
        """Get or create SQLAlchemy engine for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            SQLAlchemy engine for the project
        """
        if not self._is_valid_project_id(project_id):
            raise ValueError(f"Invalid project ID: {project_id}")
            
        if project_id not in self.engines:
            connection_string = self._get_connection_string(project_id)
            
            # Configure engine
            db_config = self.config.get('database', {})
            echo = db_config.get('echo', False)
            
            # Set up connection arguments
            connect_args = {}
            if self.db_type == 'sqlite':
                connect_args['check_same_thread'] = False
            
            # Create engine
            engine = create_engine(
                connection_string,
                echo=echo,
                future=True,
                connect_args=connect_args
            )
            
            self.engines[project_id] = engine
            self.session_factories[project_id] = sessionmaker(bind=engine)
            
            # Create tables if they don't exist
            if self.db_type == 'sqlite':
                db_path = self.db_root / f"{project_id}.sqlite"
                if not db_path.exists():
                    Base.metadata.create_all(engine)
            
            # For PostgreSQL with schemas, we would need to create the schema here
            # and set the search path
        
        return self.engines[project_id]
    
    def get_session(self, project_id: Optional[str] = None) -> Session:
        """Get a database session for the specified or current project.
        
        Args:
            project_id: Project identifier (uses current project if None)
            
        Returns:
            SQLAlchemy session for the project
            
        Raises:
            ValueError: If no project is specified and no current project is set
        """
        project = project_id or self.current_project
        if not project:
            raise ValueError("No project specified and no current project set")
        
        engine = self.get_engine(project)
        if project not in self.session_factories:
            self.session_factories[project] = sessionmaker(bind=engine)
        
        return self.session_factories[project]()
    
    def set_current_project(self, project_id: str) -> bool:
        """Set the current project context.
        
        Args:
            project_id: Project identifier
            
        Returns:
            True if the project was set successfully, False otherwise
        """
        if not self._is_valid_project_id(project_id):
            logger.error(f"Invalid project ID: {project_id}")
            return False
            
        # Create project if it doesn't exist
        if project_id not in self.get_project_list():
            if not self.create_project(project_id):
                return False
        
        # Make sure the engine is initialized
        self.get_engine(project_id)
        
        self.current_project = project_id
        return True
    
    def get_current_project(self) -> Optional[str]:
        """Get the current project ID.
        
        Returns:
            Current project ID or None if no project is set
        """
        return self.current_project


# Singleton instance
_project_db_manager = None

def get_project_db_manager() -> ProjectDatabaseManager:
    """Get the singleton instance of ProjectDatabaseManager.
    
    Returns:
        Singleton ProjectDatabaseManager instance
    """
    global _project_db_manager
    if _project_db_manager is None:
        _project_db_manager = ProjectDatabaseManager()
    return _project_db_manager

def get_session(project_id: Optional[str] = None) -> Session:
    """Get a database session for the specified or current project.
    
    Args:
        project_id: Project identifier (uses current project if None)
        
    Returns:
        SQLAlchemy session for the project
    """
    manager = get_project_db_manager()
    return manager.get_session(project_id)

@contextmanager
def session_scope(project_id: Optional[str] = None):
    """Provide a transactional scope around a series of operations.
    
    Args:
        project_id: Project identifier (uses current project if None)
        
    Yields:
        SQLAlchemy session
    """
    manager = get_project_db_manager()
    session = manager.get_session(project_id)
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()