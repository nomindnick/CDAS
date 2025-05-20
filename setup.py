from setuptools import setup, find_packages

setup(
    name="cdas",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
        "pdfplumber",
        "pandas",
        "sqlalchemy",
        "alembic",
    ],
)