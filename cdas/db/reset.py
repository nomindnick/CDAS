"""
Database reset script for the Construction Document Analysis System.

This module provides functions for resetting the database, dropping all tables,
and recreating the schema. This should only be used in development environments.
"""

import os
import argparse
import logging
import sys

from sqlalchemy.orm import Session

from cdas.db.models import Base
from cdas.db.session import get_engine, init_db
from cdas.db.init import init_database
from cdas.config import get_config


logger = logging.getLogger(__name__)


def drop_database(engine=None):
    """Drop all tables from the database.
    
    Args:
        engine: SQLAlchemy engine (defaults to the global engine if None)
    """
    if engine is None:
        engine = get_engine()
    
    logger.info("Dropping all database tables")
    Base.metadata.drop_all(engine)
    logger.info("Database tables dropped successfully")


def reset_database(engine=None):
    """Reset the database by dropping all tables and recreating them.
    
    Args:
        engine: SQLAlchemy engine (defaults to the global engine if None)
    """
    if engine is None:
        engine = get_engine()
    
    drop_database(engine)
    init_database(engine)


def main():
    """Command-line entry point for resetting the database."""
    parser = argparse.ArgumentParser(description="Reset the CDAS database")
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
        "--echo", 
        action="store_true",
        help="Echo SQL statements"
    )
    parser.add_argument(
        "--confirm", 
        action="store_true",
        help="Skip confirmation prompt"
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
    
    # Confirm reset
    if not args.confirm:
        logger.warning("This will delete all data in the database!")
        confirm = input("Are you sure you want to continue? [y/N] ")
        if confirm.lower() != "y":
            logger.info("Aborting database reset")
            return
    
    # Initialize database connection
    init_db({"database": db_config})
    
    # Reset database
    logger.info(f"Resetting {db_config['db_type']} database")
    reset_database()
    
    logger.info("Database reset complete")


if __name__ == "__main__":
    main()