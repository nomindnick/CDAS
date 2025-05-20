"""
Database initialization script for the Construction Document Analysis System.

This module provides functions for initializing the database schema and 
creating the necessary tables.
"""

import os
import argparse
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from cdas.db.models import Base
from cdas.db.session import get_engine, get_session, init_db
from cdas.config import get_config


logger = logging.getLogger(__name__)


def init_database(engine=None):
    """Initialize the database schema.
    
    Args:
        engine: SQLAlchemy engine (defaults to the global engine if None)
    """
    if engine is None:
        engine = get_engine()
    
    logger.info("Creating database tables")
    Base.metadata.create_all(engine)
    logger.info("Database tables created successfully")


def ensure_directory_exists(path):
    """Ensure that the directory for the database file exists.
    
    Args:
        path: Path to the database file
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def main():
    """Command-line entry point for initializing the database."""
    parser = argparse.ArgumentParser(description="Initialize the CDAS database")
    parser.add_argument(
        "--config", 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--db-path", 
        help="Path to database file (for SQLite)"
    )
    parser.add_argument(
        "--db-type", 
        choices=["sqlite", "postgresql"],
        help="Database type"
    )
    parser.add_argument(
        "--db-host", 
        help="Database host (for PostgreSQL)"
    )
    parser.add_argument(
        "--db-port", 
        type=int,
        help="Database port (for PostgreSQL)"
    )
    parser.add_argument(
        "--db-name", 
        help="Database name (for PostgreSQL)"
    )
    parser.add_argument(
        "--db-user", 
        help="Database user (for PostgreSQL)"
    )
    parser.add_argument(
        "--db-password", 
        help="Database password (for PostgreSQL)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force initialization even if database exists"
    )
    parser.add_argument(
        "--echo", 
        action="store_true",
        help="Echo SQL statements"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    config = get_config()
    
    # Override configuration with command-line arguments
    db_config = config.get("database", {})
    
    if args.config:
        logger.info(f"Using configuration file: {args.config}")
        os.environ["CDAS_CONFIG"] = args.config
        config = get_config()
        db_config = config.get("database", {})
    
    if args.db_type:
        db_config["db_type"] = args.db_type
    
    if args.db_path:
        db_config["db_path"] = args.db_path
    
    if args.db_host:
        db_config["db_host"] = args.db_host
    
    if args.db_port:
        db_config["db_port"] = args.db_port
    
    if args.db_name:
        db_config["db_name"] = args.db_name
    
    if args.db_user:
        db_config["db_user"] = args.db_user
    
    if args.db_password:
        db_config["db_password"] = args.db_password
    
    if args.echo:
        db_config["echo"] = True
    
    # Initialize database
    if db_config["db_type"] == "sqlite":
        db_path = db_config["db_path"]
        if os.path.exists(db_path) and not args.force:
            logger.warning(f"Database file already exists: {db_path}")
            logger.warning("Use --force to reinitialize")
            return
        
        ensure_directory_exists(db_path)
    
    logger.info(f"Initializing {db_config['db_type']} database")
    init_db({"database": db_config})
    init_database()
    
    logger.info("Database initialization complete")


if __name__ == "__main__":
    main()