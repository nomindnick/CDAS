"""
Tests specifically for mock mode operation.

Mock mode allows the system to function without actual API keys,
which is essential for testing and development. These tests ensure
that the mock mode works correctly across all AI components.
"""

import pytest
import json
from unittest import mock

from cdas.ai.llm import LLMManager
from cdas.ai.embeddings import EmbeddingManager
from cdas.ai.semantic_search.search import SemanticSearch
from cdas.ai.semantic_search.index import VectorIndex
from cdas.ai.agents.investigator import InvestigatorAgent
from cdas.ai.agents.reporter import ReporterAgent


def test_llm_manager_mock_mode(test_session):
    """Test that LLM manager works correctly in mock mode."""
    # Create LLM manager in mock mode
    config = {"mock_mode": True}
    llm = LLMManager(test_session, config)
    
    # Verify it's in mock mode
    assert llm.mock_mode
    
    # Test text generation
    result = llm.generate("Tell me about construction disputes.")
    assert "[MOCK]" in result
    assert "construction disputes" in result.lower()
    
    # Test that response is different based on input
    result2 = llm.generate("Tell me about electrical work.")
    assert "[MOCK]" in result2
    assert "electrical work" in result2.lower()
    assert result != result2
    
    # Test generation with system prompt
    result_with_system = llm.generate(
        "What are the common causes of construction disputes?",
        system_prompt="You are a construction law expert."
    )
    assert "[MOCK]" in result_with_system
    assert "construction law expert" in result_with_system.lower()
    
    # Test generation with tools
    result_with_tools = llm.generate_with_tools(
        "Analyze this invoice for $15,000.",
        system_prompt="You are a financial analyst.",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search for documents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    )
    
    # Check the structure of the response with tools
    assert "content" in result_with_tools
    assert "[MOCK]" in result_with_tools["content"]
    
    # Tool calls should be included in mock mode
    assert "tool_calls" in result_with_tools
    assert result_with_tools["tool_calls"] is not None
    
    # Check that at least one tool call is included
    assert len(result_with_tools["tool_calls"]) > 0
    assert "function" in result_with_tools["tool_calls"][0]
    assert "name" in result_with_tools["tool_calls"][0]["function"]
    
    # The tool name should match one of our provided tools
    assert result_with_tools["tool_calls"][0]["function"]["name"] == "search_documents"


def test_embedding_manager_mock_mode(test_session):
    """Test that embedding manager works correctly in mock mode."""
    # Create embedding manager in mock mode
    config = {"mock_mode": True}
    embedding_manager = EmbeddingManager(test_session, config)
    
    # Verify it's in mock mode
    assert embedding_manager.mock_mode
    
    # Test embedding generation
    embedding = embedding_manager.generate_embeddings("This is a test document.")
    
    # Verify embedding structure
    assert isinstance(embedding, list)
    assert len(embedding) == 1536  # Should match text-embedding-3-small dimensions
    
    # Verify embedding values are deterministic based on input
    embedding2 = embedding_manager.generate_embeddings("This is a test document.")
    assert embedding == embedding2  # Same input should produce same mock embedding
    
    # Verify different inputs produce different embeddings
    embedding3 = embedding_manager.generate_embeddings("This is a different document.")
    assert embedding != embedding3  # Different input should produce different mock embedding
    
    # Verify embedding batch processing
    batch_result = embedding_manager.generate_embeddings_batch(
        ["Document 1", "Document 2", "Document 3"]
    )
    
    # Check batch result structure
    assert isinstance(batch_result, list)
    assert len(batch_result) == 3
    
    # Each embedding in batch should be different
    assert batch_result[0] != batch_result[1]
    assert batch_result[1] != batch_result[2]
    assert batch_result[0] != batch_result[2]


def test_semantic_search_mock_mode(test_session):
    """Test that semantic search works correctly in mock mode."""
    # Create components in mock mode
    config = {"mock_mode": True}
    embedding_manager = EmbeddingManager(test_session, config)
    vector_index = VectorIndex()
    search = SemanticSearch(test_session, embedding_manager, vector_index)
    
    # Add some test vectors to the index
    for i in range(5):
        # In mock mode, these vectors would be generated deterministically
        vector_index.add(
            embedding_manager.generate_embeddings(f"Document {i+1}"),
            {
                "doc_id": f"doc{i+1}",
                "doc_type": "test",
                "text": f"This is document {i+1}"
            }
        )
    
    # Test semantic search
    results = search.search("test query")
    
    # Verify search results
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Each result should have a score and metadata
    for result in results:
        assert "score" in result
        assert "metadata" in result
        assert "doc_id" in result["metadata"]
        assert "doc_type" in result["metadata"]
        assert "text" in result["metadata"]
    
    # Test search with filters
    filtered_results = search.search("test query", filters={"doc_type": "test"})
    
    # Verify filtered results
    assert isinstance(filtered_results, list)
    assert len(filtered_results) > 0
    
    # All results should match the filter
    for result in filtered_results:
        assert result["metadata"]["doc_type"] == "test"


def test_investigator_agent_mock_mode(test_session):
    """Test that investigator agent works correctly in mock mode."""
    # Create LLM manager in mock mode
    config = {"mock_mode": True}
    llm = LLMManager(test_session, config)
    
    # Create investigator agent
    investigator = InvestigatorAgent(test_session, llm)
    
    # Test investigation
    results = investigator.investigate("Were there any duplicate billings?")
    
    # Verify investigation results in mock mode
    assert "findings" in results
    assert "[MOCK]" in results["findings"]
    assert "duplicate billings" in results["findings"].lower()
    
    assert "key_points" in results
    assert isinstance(results["key_points"], list)
    
    assert "confidence" in results
    assert 0 <= results["confidence"] <= 1  # Should be a confidence score between 0 and 1
    
    assert "document_ids" in results
    assert isinstance(results["document_ids"], list)
    
    # Test custom investigation
    with mock.patch.object(investigator.llm, 'generate_with_tools', 
                          return_value={
                              'content': '[MOCK] Custom investigation findings',
                              'tool_calls': [
                                  {
                                      'function': {
                                          'name': 'search_documents',
                                          'arguments': '{"query": "duplicate billing"}'
                                      }
                                  }
                              ]
                          }):
        
        # Mock tool execution
        with mock.patch.object(investigator, '_execute_tool_calls', 
                              return_value={
                                  'search_documents': json.dumps({
                                      'results': [
                                          {'doc_id': 'doc1', 'text': 'Invoice with duplicate billing'},
                                          {'doc_id': 'doc2', 'text': 'Payment application with same charges'}
                                      ]
                                  })
                              }):
            
            # Investigate with custom question in mock mode
            custom_results = investigator.investigate_with_tools(
                "Find evidence of duplicative charges",
                custom_tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "analyze_amounts",
                            "description": "Analyze amounts in documents",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "min_amount": {"type": "number"},
                                    "max_amount": {"type": "number"}
                                }
                            }
                        }
                    }
                ]
            )
            
            # Verify custom investigation results
            assert "findings" in custom_results
            assert "[MOCK]" in custom_results["findings"]
            assert custom_results["document_ids"] == ["doc1", "doc2"]
            assert "duplicative charges" in custom_results["findings"].lower()


def test_reporter_agent_mock_mode(test_session):
    """Test that reporter agent works correctly in mock mode."""
    # Create LLM manager in mock mode
    config = {"mock_mode": True}
    llm = LLMManager(test_session, config)
    
    # Create reporter agent
    reporter = ReporterAgent(test_session, llm)
    
    # Test report generation
    analysis_results = {
        "key_findings": [
            {"description": "Duplicate billing for electrical work", "amount": 15000.00},
            {"description": "Rejected change order in payment app", "amount": 25000.00}
        ],
        "suspicious_patterns": [
            {"pattern_type": "duplicate_billing", "description": "Same work billed twice"}
        ],
        "disputed_amounts": [
            {"amount": 15000.00, "description": "Electrical work"},
            {"amount": 25000.00, "description": "Foundation work"}
        ]
    }
    
    # Generate executive summary
    summary = reporter.generate_executive_summary(analysis_results)
    
    # Verify summary in mock mode
    assert "[MOCK]" in summary
    assert "executive summary" in summary.lower()
    
    # Test different report types
    report_types = [
        ("detailed_report", "attorney"),
        ("evidence_chain", (25000.00, "Foundation work", [])),
        ("presentation_report", "client"),
        ("dispute_narrative", ({}, {}))
    ]
    
    for report_type, args in report_types:
        # Get the report generation method
        report_method = getattr(reporter, f"generate_{report_type}")
        
        # Generate the report
        if isinstance(args, tuple):
            report = report_method(*args)
        else:
            report = report_method(analysis_results, args)
        
        # Verify report in mock mode
        assert "[MOCK]" in report
        assert report_type.replace("_", " ") in report.lower()
    
    # Test report generation with tools
    report_result = reporter.generate_report_with_tools("executive_summary", analysis_results)
    
    # Verify report with tools result
    assert report_result["report_type"] == "executive_summary"
    assert "[MOCK]" in report_result["report_content"]
    assert "executive summary" in report_result["report_content"].lower()
    assert "metadata" in report_result
    
    # In mock mode, tool calls should be simulated but not actually made
    assert report_result["metadata"]["tool_calls"] == 0


def test_fallback_to_mock_mode(test_session):
    """Test that the system falls back to mock mode when API keys are invalid."""
    # Test with empty API keys
    empty_config = {
        "openai_api_key": "",
        "anthropic_api_key": ""
    }
    
    # Create LLM manager with empty keys
    llm_empty = LLMManager(test_session, empty_config)
    
    # Should fall back to mock mode
    assert llm_empty.mock_mode
    
    # Test with None API keys
    none_config = {
        "openai_api_key": None,
        "anthropic_api_key": None
    }
    
    # Create LLM manager with None keys
    llm_none = LLMManager(test_session, none_config)
    
    # Should fall back to mock mode
    assert llm_none.mock_mode
    
    # Test with valid keys but explicit mock mode
    mixed_config = {
        "openai_api_key": "sk-valid-key",
        "mock_mode": True
    }
    
    # Create LLM manager with valid key but explicit mock mode
    llm_mixed = LLMManager(test_session, mixed_config)
    
    # Should respect explicit mock mode setting
    assert llm_mixed.mock_mode
    
    # Test that generated text still works in mock mode
    result = llm_mixed.generate("Test query")
    assert "[MOCK]" in result


def test_mock_mode_consistency(test_session):
    """Test that mock mode generates consistent results for the same inputs."""
    # Create LLM and embedding managers in mock mode
    config = {"mock_mode": True}
    llm1 = LLMManager(test_session, config)
    llm2 = LLMManager(test_session, config)
    embedding1 = EmbeddingManager(test_session, config)
    embedding2 = EmbeddingManager(test_session, config)
    
    # Test text consistency across instances
    text = "Test consistency in mock mode."
    result1 = llm1.generate(text)
    result2 = llm2.generate(text)
    assert result1 == result2
    
    # Test embedding consistency across instances
    emb1 = embedding1.generate_embeddings(text)
    emb2 = embedding2.generate_embeddings(text)
    assert emb1 == emb2
    
    # Test consistency with same instance but different calls
    result3 = llm1.generate(text)
    assert result1 == result3
    
    emb3 = embedding1.generate_embeddings(text)
    assert emb1 == emb3
    
    # Different inputs should produce different results
    text2 = "Different input text."
    result4 = llm1.generate(text2)
    assert result1 != result4
    
    emb4 = embedding1.generate_embeddings(text2)
    assert emb1 != emb4


if __name__ == "__main__":
    pytest.main(["-v", __file__])