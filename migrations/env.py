from logging.config import fileConfig
import os
import sys
import argparse
from pathlib import Path
import logging

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

from cdas.db.models import Base
from cdas.config import get_config as get_cdas_config
from cdas.db.project_manager import get_project_db_manager

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Create a logger for this script
logger = logging.getLogger('alembic.env')

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# Import all model modules to ensure they are registered with Base
# This is important for Alembic to detect all model changes
from cdas.db.models import (
    Document, Page, LineItem, DocumentRelationship, FinancialTransaction,
    Annotation, AnalysisFlag, Report, ReportEvidence, AmountMatch,
    ChangeOrder, PaymentApplication
)

# Parse command-line arguments for project-specific migrations
parser = argparse.ArgumentParser(description='Run database migrations')
parser.add_argument('--project', help='Project ID to migrate')
parser.add_argument('--all-projects', action='store_true', help='Migrate all projects')
parser.add_argument('--create-all', action='store_true', help='Create tables for all projects')

# Extract sys.argv elements that are for our parser
# This is needed because Alembic uses its own argument parser
our_args = []
i = 0
while i < len(sys.argv):
    if sys.argv[i] in ('--project', '--all-projects', '--create-all'):
        our_args.append(sys.argv[i])
        if sys.argv[i] == '--project' and i + 1 < len(sys.argv):
            our_args.append(sys.argv[i + 1])
            i += 1
    i += 1

# Parse our arguments
args, _ = parser.parse_known_args(our_args)

# Get the project database manager
project_manager = get_project_db_manager()


def get_url(project_id=None):
    """Get database URL from configuration.
    
    Args:
        project_id: Optional project ID to get URL for
    
    Returns:
        Database URL string
    """
    if project_id:
        # Get URL for specific project
        return project_manager._get_connection_string(project_id)
    else:
        # Get default URL
        cdas_config = get_cdas_config().get('database', {})
        db_type = cdas_config.get('db_type', 'sqlite')
        
        if db_type == 'sqlite':
            db_path = cdas_config.get('db_path', 'cdas.db')
            return f"sqlite:///{db_path}"
        elif db_type == 'postgresql':
            host = cdas_config.get('db_host', 'localhost')
            port = cdas_config.get('db_port', 5432)
            database = cdas_config.get('db_name', 'cdas')
            user = cdas_config.get('db_user', 'postgres')
            password = cdas_config.get('db_password', '')
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")


def run_migrations_for_project(project_id):
    """Run migrations for a specific project.
    
    Args:
        project_id: Project ID to migrate
    """
    logger.info(f"Running migrations for project: {project_id}")
    url = get_url(project_id)
    
    # Create a new config section for this project
    section = f"project:{project_id}"
    if not config.has_section(section):
        config.add_section(section)
    
    # Set the URL in the project section
    config.set_section_option(section, "sqlalchemy.url", url)
    
    # Create engine for this project
    project_settings = dict(config.get_section(section))
    project_settings.update({
        'sqlalchemy.url': url
    })
    
    connectable = engine_from_config(
        project_settings,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    
    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata,
            version_table=f"alembic_version"
        )
        
        with context.begin_transaction():
            context.run_migrations()


def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    # Check for project-specific migration
    if args.project:
        url = get_url(args.project)
        logger.info(f"Running offline migrations for project: {args.project}")
    else:
        # Default behavior
        url = get_url()
        logger.info("Running offline migrations with default database")
    
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # Check if we're migrating a specific project
    if args.project:
        run_migrations_for_project(args.project)
    
    # Check if we should migrate all projects
    elif args.all_projects:
        projects = project_manager.get_project_list()
        if not projects:
            logger.info("No projects found to migrate")
            return
        
        for project_id in projects:
            run_migrations_for_project(project_id)
    
    # Check if we should just create tables for all projects
    elif args.create_all:
        projects = project_manager.get_project_list()
        if not projects:
            logger.info("No projects found to create tables for")
            return
        
        for project_id in projects:
            logger.info(f"Creating tables for project: {project_id}")
            engine = project_manager.get_engine(project_id)
            Base.metadata.create_all(engine)
    
    # Default behavior - just migrate the main database
    else:
        # Override sqlalchemy.url in alembic.ini
        url = get_url()
        config.set_main_option("sqlalchemy.url", url)
        logger.info("Running migrations with default database")
        
        connectable = engine_from_config(
            config.get_section(config.config_ini_section),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

        with connectable.connect() as connection:
            context.configure(
                connection=connection, target_metadata=target_metadata
            )

            with context.begin_transaction():
                context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()