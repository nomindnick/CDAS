"""
Integration tests for the semantic search capabilities.

Tests the integration between VectorIndex, Vectorizer, EmbeddingManager,
and SemanticSearch components.
"""

import pytest
from unittest import mock

from cdas.ai.semantic_search.search import SemanticSearch
from cdas.ai.semantic_search.index import VectorIndex
from cdas.ai.semantic_search.vectorizer import Vectorizer
from cdas.ai.embeddings import EmbeddingManager

from .test_helpers import (
    create_test_document,
    create_test_documents_with_amounts,
    setup_mock_embedding_response,
    verify_vector_similarity
)


def test_document_vectorization_and_search(test_session, embedding_manager, 
                                          vector_index, mock_openai):
    """Test end-to-end document vectorization and semantic search."""
    # Create vectorizer
    vectorizer = Vectorizer(embedding_manager)
    
    # Create semantic search
    search = SemanticSearch(test_session, embedding_manager, vector_index)
    
    # Create test documents
    doc1_content = """
    Invoice #123
    ABC Construction Company
    
    Item 1: Electrical work phase 1 - $15,000
    Item 2: Plumbing fixtures - $8,000
    
    Total: $23,000
    """
    
    doc2_content = """
    Change Order #7
    ABC Construction Company
    
    Additional foundation work required due to soil conditions.
    Amount: $25,000
    
    Status: Rejected
    """
    
    doc3_content = """
    Payment Application #4
    ABC Construction Company
    
    Previously completed work:
    1. Electrical work phase 1 - $15,000
    2. Plumbing fixtures - $8,000
    3. Foundation work - $25,000
    
    Total this application: $48,000
    """
    
    # Create documents in test database
    doc1 = create_test_document(test_session, doc1_content, "invoice")
    doc2 = create_test_document(test_session, doc2_content, "change_order")
    doc3 = create_test_document(test_session, doc3_content, "payment_app")
    
    # Set up mock embedding responses for each document
    mock_embeddings = [
        [0.1, 0.2, 0.3] * 512,  # Invoice embedding
        [0.2, 0.3, 0.4] * 512,  # Change order embedding
        [0.3, 0.4, 0.5] * 512,  # Payment application embedding
        [0.15, 0.25, 0.35] * 512,  # Query about electrical work
        [0.25, 0.35, 0.45] * 512,  # Query about foundation work
    ]
    
    setup_mock_embedding_response(
        mock_openai,
        [doc1_content, doc2_content, doc3_content, 
         "electrical work", "foundation work"],
        mock_embeddings
    )
    
    # Vectorize documents and add to index
    for doc, content in [(doc1, doc1_content), (doc2, doc2_content), (doc3, doc3_content)]:
        # Mock the document retrieval
        with mock.patch('cdas.ai.semantic_search.vectorizer.get_document_text', 
                       return_value=content):
            chunks = vectorizer.vectorize_document({
                "doc_id": doc.doc_id,
                "doc_type": doc.doc_type,
                "content": content
            })
            
            # Add chunks to index
            for chunk in chunks:
                vector_index.add(chunk["vector"], chunk["metadata"])
    
    # Test semantic search for electrical work
    results_electrical = search.search("electrical work")
    
    # Verify results
    assert len(results_electrical) > 0
    found_invoice = False
    found_payment_app = False
    
    for result in results_electrical:
        if result["metadata"]["doc_id"] == doc1.doc_id:
            found_invoice = True
        if result["metadata"]["doc_id"] == doc3.doc_id:
            found_payment_app = True
    
    assert found_invoice, "Invoice should be found for 'electrical work' query"
    assert found_payment_app, "Payment application should be found for 'electrical work' query"
    
    # Test semantic search for foundation work
    results_foundation = search.search("foundation work")
    
    # Verify results
    assert len(results_foundation) > 0
    found_change_order = False
    found_payment_app = False
    
    for result in results_foundation:
        if result["metadata"]["doc_id"] == doc2.doc_id:
            found_change_order = True
        if result["metadata"]["doc_id"] == doc3.doc_id:
            found_payment_app = True
    
    assert found_change_order, "Change order should be found for 'foundation work' query"
    assert found_payment_app, "Payment application should be found for 'foundation work' query"


def test_filtered_semantic_search(test_session, embedding_manager, vector_index, mock_openai):
    """Test semantic search with metadata filters."""
    # Create vectorizer and search
    vectorizer = Vectorizer(embedding_manager)
    search = SemanticSearch(test_session, embedding_manager, vector_index)
    
    # Create test documents with specific metadata
    doc_data = [
        {
            "doc_type": "invoice",
            "party": "contractor",
            "content": "Invoice for electrical work phase 1 - $15,000",
            "amounts": [
                {"amount": 15000.00, "description": "Electrical work phase 1"}
            ]
        },
        {
            "doc_type": "invoice",
            "party": "subcontractor",
            "content": "Invoice for electrical materials - $7,500",
            "amounts": [
                {"amount": 7500.00, "description": "Electrical materials"}
            ]
        },
        {
            "doc_type": "change_order",
            "party": "contractor",
            "content": "Change order for additional electrical work - $8,000",
            "amounts": [
                {"amount": 8000.00, "description": "Additional electrical work"}
            ]
        }
    ]
    
    # Create documents in test database
    docs = create_test_documents_with_amounts(test_session, doc_data)
    
    # Set up mock embedding responses
    setup_mock_embedding_response(
        mock_openai,
        [d["content"] for d in doc_data] + ["electrical"],
        [
            [0.1, 0.2, 0.3] * 512,  # Invoice 1
            [0.2, 0.3, 0.4] * 512,  # Invoice 2
            [0.3, 0.4, 0.5] * 512,  # Change order
            [0.2, 0.3, 0.4] * 512,  # Query "electrical"
        ]
    )
    
    # Vectorize documents and add to index
    for i, doc in enumerate(docs):
        # Mock the document retrieval
        with mock.patch('cdas.ai.semantic_search.vectorizer.get_document_text', 
                       return_value=doc_data[i]["content"]):
            chunks = vectorizer.vectorize_document({
                "doc_id": doc.doc_id,
                "doc_type": doc.doc_type,
                "party": doc_data[i]["party"],
                "content": doc_data[i]["content"]
            })
            
            # Add chunks to index
            for chunk in chunks:
                vector_index.add(chunk["vector"], chunk["metadata"])
    
    # Test unfiltered search
    results_all = search.search("electrical")
    assert len(results_all) == 3, "Unfiltered search should return all 3 documents"
    
    # Test filtered search by doc_type
    results_invoices = search.search("electrical", filters={"doc_type": "invoice"})
    assert len(results_invoices) == 2, "Filtered search should return 2 invoices"
    
    # Test filtered search by party
    results_contractor = search.search("electrical", filters={"party": "contractor"})
    assert len(results_contractor) == 2, "Filtered search should return 2 contractor documents"
    
    # Test combined filters
    results_combined = search.search(
        "electrical", 
        filters={"doc_type": "invoice", "party": "subcontractor"}
    )
    assert len(results_combined) == 1, "Combined filters should return 1 document"


def test_hybrid_search(test_session, embedding_manager, vector_index, mock_openai):
    """Test hybrid search combining semantic and keyword search."""
    # Create vectorizer and search
    vectorizer = Vectorizer(embedding_manager)
    search = SemanticSearch(test_session, embedding_manager, vector_index)
    
    # Create test documents with various contents
    docs_content = [
        "Invoice #123 for electrical installation work - $12,500",
        "Change order for additional electrical outlets in classroom A - $2,500",
        "Payment application including electrical work and plumbing - $35,000",
        "Memo regarding delay in electrical work completion",
        "Contract amendment for extended project timeline"
    ]
    
    # Create documents in test database
    docs = []
    doc_types = ["invoice", "change_order", "payment_app", "correspondence", "contract"]
    for i, content in enumerate(docs_content):
        doc = create_test_document(test_session, content, doc_types[i])
        docs.append(doc)
    
    # Set up mock embedding responses
    doc_embeddings = [
        [0.1, 0.2, 0.3] * 512,
        [0.2, 0.3, 0.4] * 512,
        [0.3, 0.4, 0.5] * 512,
        [0.4, 0.5, 0.6] * 512,
        [0.5, 0.6, 0.7] * 512,
    ]
    query_embedding = [0.25, 0.35, 0.45] * 512  # Query about electrical work and cost
    
    setup_mock_embedding_response(
        mock_openai,
        docs_content + ["electrical work cost"],
        doc_embeddings + [query_embedding]
    )
    
    # Vectorize documents and add to index
    for i, doc in enumerate(docs):
        # Mock the document retrieval
        with mock.patch('cdas.ai.semantic_search.vectorizer.get_document_text', 
                       return_value=docs_content[i]):
            chunks = vectorizer.vectorize_document({
                "doc_id": doc.doc_id,
                "doc_type": doc_types[i],
                "content": docs_content[i]
            })
            
            # Add chunks to index
            for chunk in chunks:
                vector_index.add(chunk["vector"], chunk["metadata"])
    
    # Set up mock for keyword search results
    with mock.patch('cdas.ai.semantic_search.search.keyword_search') as mock_keyword_search:
        # Mock keyword search to return the first 3 documents (they contain "electrical")
        mock_keyword_search.return_value = [
            {"doc_id": docs[0].doc_id, "score": 0.9, "text": docs_content[0]},
            {"doc_id": docs[1].doc_id, "score": 0.8, "text": docs_content[1]},
            {"doc_id": docs[2].doc_id, "score": 0.7, "text": docs_content[2]},
            {"doc_id": docs[3].doc_id, "score": 0.6, "text": docs_content[3]},
        ]
        
        # Test hybrid search
        results = search.hybrid_search("electrical work cost")
        
        # Verify results
        assert len(results) >= 4, "Hybrid search should return at least 4 documents"
        
        # First result should be more relevant to both semantic and keyword aspects
        assert results[0]["score"] > 0.7, "Top result should have high relevance score"
        
        # Check that results are properly deduplicated and merged
        doc_ids = [r["doc_id"] for r in results]
        assert len(doc_ids) == len(set(doc_ids)), "Results should have no duplicates"


def test_semantic_query_interface(test_session, embedding_manager, vector_index, mock_openai):
    """Test the semantic_query interface that provides a unified search experience."""
    # Set up mock for semantic search and hybrid search
    with mock.patch('cdas.ai.semantic_search.search.SemanticSearch.search') as mock_semantic_search, \
         mock.patch('cdas.ai.semantic_search.search.SemanticSearch.hybrid_search') as mock_hybrid_search:
         
        # Mock semantic search to return some results
        mock_semantic_search.return_value = [
            {"doc_id": "doc1", "score": 0.9, "text": "Document 1"},
            {"doc_id": "doc2", "score": 0.8, "text": "Document 2"}
        ]
        
        # Mock hybrid search to return some results
        mock_hybrid_search.return_value = [
            {"doc_id": "doc1", "score": 0.95, "text": "Document 1"},
            {"doc_id": "doc2", "score": 0.85, "text": "Document 2"},
            {"doc_id": "doc3", "score": 0.75, "text": "Document 3"}
        ]
        
        # Call semantic_query with default (hybrid) mode
        from cdas.ai.semantic_search.search import semantic_query
        results_hybrid = semantic_query(
            test_session, 
            embedding_manager, 
            "test query", 
            filters={"doc_type": "invoice"},
            use_hybrid=True
        )
        
        # Call semantic_query with semantic-only mode
        results_semantic = semantic_query(
            test_session,
            embedding_manager,
            "test query",
            filters={"doc_type": "invoice"},
            use_hybrid=False
        )
        
        # Verify correct method was called
        mock_hybrid_search.assert_called_once()
        mock_semantic_search.assert_called_once()
        
        # Verify results
        assert len(results_hybrid) == 3, "Hybrid search should return 3 documents"
        assert len(results_semantic) == 2, "Semantic search should return 2 documents"


def test_structure_aware_chunking(embedding_manager, mock_openai):
    """Test that vectorizer correctly chunks documents while preserving structure."""
    # Create vectorizer
    vectorizer = Vectorizer(embedding_manager)
    
    # Document with structure (headers, paragraphs, tables)
    structured_doc = """
    # Invoice #123
    
    Date: 2023-05-15
    Contractor: ABC Construction
    
    ## Line Items
    
    1. Site preparation work - $45,000
    2. Foundation work:
       - Materials: $30,000
       - Labor: $25,000
    3. Electrical rough-in - $35,000
    
    ## Summary
    
    Total amount: $135,000
    
    ## Terms
    
    Payment due within 30 days.
    """
    
    # Set up mock embedding for each chunk
    setup_mock_embedding_response(
        mock_openai,
        ["This is a chunk"], 
        [[0.1, 0.2, 0.3] * 512]
    )
    
    # Vectorize the document
    with mock.patch('cdas.ai.semantic_search.vectorizer.get_document_text', 
                   return_value=structured_doc):
        chunks = vectorizer._chunk_text(structured_doc)
        
        # Check that structural elements are preserved
        headers_preserved = 0
        for chunk in chunks:
            if "# Invoice #123" in chunk:
                headers_preserved += 1
            if "## Line Items" in chunk:
                headers_preserved += 1
            if "## Summary" in chunk:
                headers_preserved += 1
            if "## Terms" in chunk:
                headers_preserved += 1
        
        # Verify chunks preserve structure
        assert headers_preserved >= 3, "Chunking should preserve most headers"
        
        # Check that related content stays together
        for chunk in chunks:
            if "Total amount: $135,000" in chunk:
                assert "## Summary" in chunk, "Summary header should stay with total amount"
            
            if "Foundation work:" in chunk:
                assert "Materials: $30,000" in chunk, "Related items should stay together"


def test_mock_mode_for_embeddings(test_session):
    """Test that embedding manager operates correctly in mock mode."""
    # Create embedding manager in mock mode
    config = {"mock_mode": True}
    embedding_manager = EmbeddingManager(test_session, config)
    
    # Generate embeddings in mock mode
    embedding = embedding_manager.generate_embeddings("Test text")
    
    # Verify embedding is a mock embedding
    assert len(embedding) == 1536, "Mock embedding should have 1536 dimensions"
    assert embedding[0] == 0.1, "First dimension of mock embedding should be 0.1"
    
    # Verify consistent mock embeddings for the same text
    embedding2 = embedding_manager.generate_embeddings("Test text")
    assert embedding == embedding2, "Mock embeddings should be consistent for the same text"
    
    # Verify different mock embeddings for different text
    embedding3 = embedding_manager.generate_embeddings("Different text")
    assert embedding != embedding3, "Mock embeddings should be different for different text"


if __name__ == "__main__":
    pytest.main(["-v", __file__])