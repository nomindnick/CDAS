"""
Pytest fixtures for AI integration tests.

These fixtures set up a test environment for AI component integration testing,
including LLM, embeddings, and semantic search components.
"""

import os
import json
import pytest
from unittest import mock

from cdas.ai.llm import LLMManager
from cdas.ai.embeddings import EmbeddingManager
from cdas.ai.semantic_search.index import VectorIndex
from cdas.ai.semantic_search.vectorizer import Vectorizer
from cdas.ai.semantic_search import search
from cdas.ai.agents.investigator import InvestigatorAgent
from cdas.ai.agents.reporter import ReporterAgent
from cdas.ai.tools import document_tools
from cdas.ai.tools import financial_tools
from cdas.ai.tools import database_tools


# AI Component fixtures
@pytest.fixture(scope="function")
def mock_openai():
    """Mock OpenAI client for testing."""
    with mock.patch("cdas.ai.llm.OpenAI") as mock_openai:
        # Setup mock responses for various endpoints
        mock_client = mock.Mock()
        mock_openai.return_value = mock_client
        
        # Mock chat completion response
        mock_chat = mock.Mock()
        mock_message = mock.Mock()
        mock_message.content = "This is a test response"
        mock_chat.choices = [mock.Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_chat
        
        # Mock embedding response
        mock_embedding_data = mock.Mock()
        mock_embedding_data.embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        mock_embedding_response = mock.Mock()
        mock_embedding_response.data = [mock_embedding_data]
        mock_client.embeddings.create.return_value = mock_embedding_response
        
        yield mock_client


@pytest.fixture(scope="function")
def mock_anthropic():
    """Mock Anthropic client for testing."""
    with mock.patch("cdas.ai.llm.Anthropic") as mock_anthropic:
        # Setup mock responses
        mock_client = mock.Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock completion response
        mock_message = mock.Mock()
        mock_message.content = [mock.Mock(text="This is a test response")]
        mock_client.messages.create.return_value = mock_message
        
        yield mock_client


@pytest.fixture(scope="function")
def llm_manager(test_session, mock_openai, mock_anthropic):
    """Create an LLM manager for testing."""
    # Create with mock mode enabled for safe testing
    config = {
        "mock_mode": True,
        "openai_model": "gpt-4o",
        "anthropic_model": "claude-3-sonnet-20240229"
    }
    
    llm = LLMManager(test_session, config)
    return llm


@pytest.fixture(scope="function")
def embedding_manager(test_session, mock_openai):
    """Create an embedding manager for testing."""
    # Create with mock mode enabled for safe testing
    config = {
        "mock_mode": True,
        "embedding_model": "text-embedding-3-small"
    }
    
    embedding_manager = EmbeddingManager(test_session, config)
    return embedding_manager


@pytest.fixture(scope="function")
def vector_index():
    """Create a vector index for testing."""
    # Create an empty vector index
    index = VectorIndex()
    
    # Add some test vectors
    test_vectors = [
        {
            "vector": [0.1, 0.2, 0.3] * 512,  # 1536 dimensions
            "metadata": {
                "doc_id": "doc1",
                "chunk_id": "chunk1",
                "doc_type": "invoice",
                "text": "This is a test invoice for $5000 for electrical work"
            }
        },
        {
            "vector": [0.2, 0.3, 0.4] * 512,
            "metadata": {
                "doc_id": "doc2",
                "chunk_id": "chunk2",
                "doc_type": "payment_app",
                "text": "Payment application for electrical work phase 1, $5000"
            }
        },
        {
            "vector": [0.3, 0.4, 0.5] * 512,
            "metadata": {
                "doc_id": "doc3",
                "chunk_id": "chunk3",
                "doc_type": "change_order",
                "text": "Change order for foundation work, additional $25000"
            }
        }
    ]
    
    for item in test_vectors:
        index.add(item["vector"], item["metadata"])
    
    return index


@pytest.fixture(scope="function")
def vectorizer(embedding_manager):
    """Create a vectorizer for testing."""
    vectorizer = Vectorizer(embedding_manager)
    return vectorizer


@pytest.fixture(scope="function")
def semantic_search(test_session, embedding_manager):
    """Create a semantic search function wrapper for testing."""
    # Return a wrapper function that delegates to the search module functions
    class SearchWrapper:
        def __init__(self, session, embedding_mgr):
            self.session = session
            self.embedding_manager = embedding_mgr
            
        def search(self, query, **kwargs):
            return search.semantic_search(self.session, self.embedding_manager, query, **kwargs)
            
        def batch_embed_documents(self, **kwargs):
            return search.batch_embed_documents(self.session, self.embedding_manager, **kwargs)
            
        def semantic_query(self, query, **kwargs):
            return search.semantic_query(self.session, self.embedding_manager, query, **kwargs)
    
    return SearchWrapper(test_session, embedding_manager)


@pytest.fixture(scope="function")
def investigator_agent(test_session, llm_manager):
    """Create an investigator agent for testing."""
    agent = InvestigatorAgent(test_session, llm_manager)
    return agent


@pytest.fixture(scope="function")
def reporter_agent(test_session, llm_manager):
    """Create a reporter agent for testing."""
    agent = ReporterAgent(test_session, llm_manager)
    return agent


@pytest.fixture(scope="function")
def document_tools(test_session):
    """Create document tools for testing."""
    # Create a wrapper for document tools functions
    class DocumentToolsWrapper:
        def __init__(self, session):
            self.session = session
            
        def search_documents(self, query, **kwargs):
            return document_tools.search_documents(self.session, query, **kwargs)
            
        def get_document_content(self, doc_id):
            return document_tools.get_document_content(self.session, doc_id)
            
        def compare_documents(self, doc_id1, doc_id2):
            return document_tools.compare_documents(self.session, doc_id1, doc_id2)
    
    return DocumentToolsWrapper(test_session)


@pytest.fixture(scope="function")
def financial_tools(test_session):
    """Create financial tools for testing."""
    # Create a wrapper for financial tools functions
    class FinancialToolsWrapper:
        def __init__(self, session):
            self.session = session
            
        def search_line_items(self, **kwargs):
            return financial_tools.search_line_items(self.session, **kwargs)
            
        def find_similar_amounts(self, amount, tolerance_percentage=0.1):
            return financial_tools.find_similar_amounts(self.session, amount, tolerance_percentage)
            
        def find_suspicious_patterns(self, pattern_type=None, min_confidence=0.5):
            return financial_tools.find_suspicious_patterns(self.session, pattern_type, min_confidence)
            
        def analyze_amount(self, amount):
            return financial_tools.analyze_amount(self.session, amount)
    
    return FinancialToolsWrapper(test_session)


@pytest.fixture(scope="function")
def database_tools(test_session):
    """Create database tools for testing."""
    # Create a wrapper for database tools functions
    class DatabaseToolsWrapper:
        def __init__(self, session):
            self.session = session
            
        def run_sql_query(self, query, params=None):
            return database_tools.run_sql_query(self.session, query, params)
            
        def get_document_relationships(self, doc_id):
            return database_tools.get_document_relationships(self.session, doc_id)
            
        def get_document_metadata(self, doc_id):
            return database_tools.get_document_metadata(self.session, doc_id)
            
        def get_amount_references(self, amount, tolerance=0.01):
            return database_tools.get_amount_references(self.session, amount, tolerance)
            
        def find_date_range_activity(self, start_date, end_date, doc_type=None, party=None):
            return database_tools.find_date_range_activity(self.session, start_date, end_date, doc_type, party)
            
        def get_document_changes(self, doc_id):
            return database_tools.get_document_changes(self.session, doc_id)
            
        def get_financial_transactions(self, **kwargs):
            return database_tools.get_financial_transactions(self.session, **kwargs)
    
    return DatabaseToolsWrapper(test_session)


@pytest.fixture(scope="function")
def sample_analysis_results():
    """Create sample analysis results for testing reports."""
    return {
        "key_findings": [
            {
                "description": "Duplicate billing for electrical work",
                "amount": 15000.00,
                "confidence": 0.92,
                "document_ids": ["doc1", "doc2"],
                "evidence": "Same work billed in both invoice #123 and payment application #4"
            },
            {
                "description": "Unapproved change order included in payment application",
                "amount": 25000.00,
                "confidence": 0.89,
                "document_ids": ["doc3", "doc4"],
                "evidence": "Change order #7 was rejected but included in payment application #5"
            }
        ],
        "suspicious_patterns": [
            {
                "pattern_type": "duplicate_billing",
                "description": "Electrical work phase 1 billed twice",
                "severity": "high",
                "document_ids": ["doc1", "doc2"],
                "financial_impact": 15000.00
            },
            {
                "pattern_type": "rejected_items_rebilled",
                "description": "Rejected change order items appearing in payment applications",
                "severity": "high",
                "document_ids": ["doc3", "doc4"],
                "financial_impact": 25000.00
            }
        ],
        "disputed_amounts": [
            {
                "amount": 15000.00,
                "description": "Electrical work phase 1",
                "status": "disputed",
                "reason": "Duplicate billing",
                "document_ids": ["doc1", "doc2"]
            },
            {
                "amount": 25000.00,
                "description": "Additional foundation work",
                "status": "disputed",
                "reason": "Change order not approved",
                "document_ids": ["doc3", "doc4"]
            },
            {
                "amount": 10000.00,
                "description": "Material price increases",
                "status": "disputed",
                "reason": "No supporting documentation",
                "document_ids": ["doc5"]
            }
        ]
    }


@pytest.fixture(scope="function")
def sample_document_trail():
    """Create sample document trail for testing evidence chains."""
    return [
        {
            "doc_id": "doc1",
            "doc_type": "contract",
            "title": "Original Contract",
            "date": "2023-01-15",
            "party": "contractor",
            "status": "active",
            "total_amount": 500000.00,
            "relevant_text": "Electrical work phase 1 to be completed for $15,000",
            "page_number": 8,
            "file_path": "/path/to/contract.pdf"
        },
        {
            "doc_id": "doc2",
            "doc_type": "invoice",
            "title": "Invoice #123",
            "date": "2023-02-20",
            "party": "contractor",
            "status": "submitted",
            "total_amount": 75000.00,
            "relevant_text": "Electrical work phase 1 - $15,000",
            "page_number": 1,
            "file_path": "/path/to/invoice123.pdf"
        },
        {
            "doc_id": "doc4",
            "doc_type": "payment_app",
            "title": "Payment Application #4",
            "date": "2023-03-10",
            "party": "contractor",
            "status": "approved",
            "total_amount": 120000.00,
            "relevant_text": "Electrical work phase 1 - $15,000",
            "page_number": 2,
            "file_path": "/path/to/payapp4.pdf"
        }
    ]


@pytest.fixture(scope="function")
def sample_project_info():
    """Create sample project information for testing."""
    return {
        "name": "Oak Elementary School Renovation",
        "type": "School Construction",
        "description": "Renovation of existing elementary school including new classroom wing and modernized facilities",
        "owner": "Woodland School District",
        "contractor": "ABC Construction Inc.",
        "contract_amount": 7500000.00,
        "start_date": "2023-01-15",
        "end_date": "2024-06-30",
        "current_status": "In Progress",
        "events": [
            {"date": "2023-01-15", "description": "Contract signed", "type": "milestone"},
            {"date": "2023-02-10", "description": "Site mobilization", "type": "milestone"},
            {"date": "2023-03-05", "description": "Change order #1 submitted - Additional foundation work", "type": "change"},
            {"date": "2023-03-15", "description": "Change order #1 rejected", "type": "change"},
            {"date": "2023-04-10", "description": "Payment application #1 submitted and approved", "type": "payment"},
            {"date": "2023-05-12", "description": "Change order items reappear in payment application #2", "type": "issue"}
        ]
    }