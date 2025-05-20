from setuptools import setup, find_packages

setup(
    name="cdas",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        "sqlalchemy",
        "alembic",
        "click",
        "pydantic",
        
        # Document processing
        "pdfplumber",
        "pandas",
        "openpyxl",
        
        # Reporting
        "weasyprint",
        "markdown",
        "jinja2",
    ],
)