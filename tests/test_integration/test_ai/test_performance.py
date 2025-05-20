"""
Performance benchmarks for AI components.

These benchmarks measure the performance of various AI components
to help identify bottlenecks and track improvements over time.
Tests are marked as "benchmark" to exclude them from normal test runs.
"""

import pytest
import time
import json
import random
import string
from unittest import mock

from cdas.ai.semantic_search.index import VectorIndex
from cdas.ai.semantic_search.vectorizer import Vectorizer
from cdas.ai.semantic_search.search import SemanticSearch
from cdas.ai.embeddings import EmbeddingManager


@pytest.mark.benchmark
def test_vector_index_performance(test_session):
    """Benchmark the performance of vector index operations."""
    # Create a vector index
    index = VectorIndex()
    
    # Generate test vectors
    num_vectors = 1000
    vector_dim = 1536
    
    vectors = []
    for i in range(num_vectors):
        # Create a random vector
        vector = [random.random() for _ in range(vector_dim)]
        metadata = {
            "doc_id": f"doc{i}",
            "chunk_id": f"chunk{i}",
            "text": f"This is test document {i}",
            "doc_type": random.choice(["invoice", "change_order", "payment_app", "contract"]),
            "party": random.choice(["contractor", "owner", "subcontractor"])
        }
        vectors.append((vector, metadata))
    
    # Benchmark addition performance
    start_time = time.time()
    for vector, metadata in vectors:
        index.add(vector, metadata)
    add_time = time.time() - start_time
    
    # Display performance metrics
    print(f"\nVector Index Addition Performance:")
    print(f"Added {num_vectors} vectors in {add_time:.4f} seconds")
    print(f"Average time per vector: {(add_time / num_vectors) * 1000:.4f} ms")
    
    # Benchmark search performance
    query_vectors = []
    for _ in range(10):
        # Create random query vectors
        query_vector = [random.random() for _ in range(vector_dim)]
        query_vectors.append(query_vector)
    
    # Warm-up search
    _ = index.search(query_vectors[0], limit=10)
    
    # Benchmark search without filters
    start_time = time.time()
    for query_vector in query_vectors:
        results = index.search(query_vector, limit=10)
        assert len(results) <= 10
    search_time = time.time() - start_time
    
    # Display performance metrics
    print(f"\nVector Index Search Performance (without filters):")
    print(f"Executed {len(query_vectors)} searches in {search_time:.4f} seconds")
    print(f"Average time per search: {(search_time / len(query_vectors)) * 1000:.4f} ms")
    
    # Benchmark search with filters
    filters = [
        {"doc_type": "invoice"},
        {"doc_type": "change_order"},
        {"party": "contractor"},
        {"doc_type": "invoice", "party": "contractor"},
        {"doc_type": "payment_app", "party": "owner"}
    ]
    
    start_time = time.time()
    for i, query_vector in enumerate(query_vectors[:5]):
        filter_dict = filters[i]
        results = index.filtered_search(query_vector, filter_dict, limit=10)
        assert len(results) <= 10
    filtered_search_time = time.time() - start_time
    
    # Display performance metrics
    print(f"\nVector Index Filtered Search Performance:")
    print(f"Executed {len(filters)} filtered searches in {filtered_search_time:.4f} seconds")
    print(f"Average time per filtered search: {(filtered_search_time / len(filters)) * 1000:.4f} ms")
    
    # Performance should be acceptable
    assert add_time / num_vectors < 0.001  # Less than 1ms per addition
    assert search_time / len(query_vectors) < 0.05  # Less than 50ms per search
    assert filtered_search_time / len(filters) < 0.05  # Less than 50ms per filtered search


@pytest.mark.benchmark
def test_vectorizer_performance(test_session, embedding_manager):
    """Benchmark the performance of document vectorization."""
    # Create a vectorizer
    vectorizer = Vectorizer(embedding_manager)
    
    # Generate test documents of various sizes
    doc_sizes = [
        ("small", 500),      # ~500 characters
        ("medium", 5000),    # ~5000 characters
        ("large", 20000),    # ~20000 characters
        ("very_large", 50000)  # ~50000 characters
    ]
    
    documents = []
    for size_name, size in doc_sizes:
        # Generate random text
        text = ''.join(random.choice(string.ascii_letters + ' \n.,:;-') for _ in range(size))
        
        # Format as a document with some structure
        doc_text = f"""
        # Test Document ({size_name})
        
        ## Introduction
        
        This is a test document of {size_name} size.
        
        ## Content
        
        {text}
        
        ## Conclusion
        
        End of test document.
        """
        
        doc = {
            "doc_id": f"doc_{size_name}",
            "doc_type": "test",
            "content": doc_text
        }
        
        documents.append((size_name, doc))
    
    # Mock embedding generation to isolate vectorizer performance
    with mock.patch.object(embedding_manager, 'generate_embeddings') as mock_generate:
        # Return a consistent mock embedding
        mock_generate.return_value = [0.1, 0.2, 0.3] * 512
        
        # Test chunking performance
        print("\nVectorizer Chunking Performance:")
        for size_name, doc in documents:
            start_time = time.time()
            chunks = vectorizer._chunk_text(doc["content"])
            chunk_time = time.time() - start_time
            
            print(f"{size_name} document ({len(doc['content'])} chars):")
            print(f"  - Created {len(chunks)} chunks in {chunk_time:.4f} seconds")
            print(f"  - Average size per chunk: {sum(len(c) for c in chunks) / len(chunks):.1f} chars")
        
        # Test full vectorization performance
        print("\nVectorizer Full Process Performance:")
        for size_name, doc in documents:
            with mock.patch('cdas.ai.semantic_search.vectorizer.get_document_text', 
                           return_value=doc["content"]):
                start_time = time.time()
                chunks = vectorizer.vectorize_document(doc)
                process_time = time.time() - start_time
                
                print(f"{size_name} document ({len(doc['content'])} chars):")
                print(f"  - Processed into {len(chunks)} vector chunks in {process_time:.4f} seconds")
                print(f"  - Average time per chunk: {(process_time / len(chunks)) * 1000:.4f} ms")
    
    # Performance assertions
    # These will need to be adjusted based on actual performance measurements
    assert True  # Placeholder for actual performance thresholds


@pytest.mark.benchmark
def test_semantic_search_performance(test_session, embedding_manager):
    """Benchmark the performance of semantic search operations."""
    # Create components
    vector_index = VectorIndex()
    search = SemanticSearch(test_session, embedding_manager, vector_index)
    
    # Generate test vectors and add to index
    num_vectors = 5000
    vector_dim = 1536
    
    # Mock embedding generation for consistent results
    with mock.patch.object(embedding_manager, 'generate_embeddings') as mock_generate:
        # Set up mock to return different vectors for different texts
        def mock_embedding_generator(text):
            """Generate deterministic but unique mock embeddings based on text."""
            # Use hash of text to seed the random generator for consistency
            random.seed(hash(text) % 10000)
            return [random.random() for _ in range(vector_dim)]
        
        mock_generate.side_effect = mock_embedding_generator
        
        # Add test vectors to index
        doc_types = ["invoice", "change_order", "payment_app", "contract", "correspondence"]
        parties = ["contractor", "owner", "subcontractor", "architect", "consultant"]
        
        print("\nPopulating vector index for search benchmark...")
        start_time = time.time()
        
        for i in range(num_vectors):
            doc_type = doc_types[i % len(doc_types)]
            party = parties[i % len(parties)]
            
            # Create text with meaningful content for better search testing
            text = f"This is document {i} of type {doc_type} from {party}. "
            
            if doc_type == "invoice":
                text += f"Invoice for construction work in the amount of ${(i % 10) * 10000 + 5000}. "
                text += "Includes labor, materials, and equipment for project phase 1."
            elif doc_type == "change_order":
                text += f"Change order for additional work in the amount of ${(i % 5) * 5000 + 1000}. "
                text += "Change required due to unforeseen site conditions."
            elif doc_type == "payment_app":
                text += f"Payment application for work completed to date: ${(i % 8) * 25000 + 10000}. "
                text += "Application covers work from previous month."
            elif doc_type == "contract":
                text += f"Contract for project with total value of ${(i % 3) * 1000000 + 500000}. "
                text += "Scope includes new construction and renovation work."
            else:
                text += f"Correspondence regarding project schedule and budget concerns. "
                text += "Requesting meeting to discuss timeline adjustments."
            
            # Generate embedding for this text
            vector = embedding_manager.generate_embeddings(text)
            
            # Add to index
            vector_index.add(vector, {
                "doc_id": f"doc{i}",
                "chunk_id": f"chunk{i}",
                "doc_type": doc_type,
                "party": party,
                "text": text
            })
        
        index_time = time.time() - start_time
        print(f"Added {num_vectors} vectors in {index_time:.4f} seconds")
        
        # Benchmark search performance with various queries
        queries = [
            "invoice for electrical work",
            "change order foundation work",
            "payment application concrete",
            "contract scope of work",
            "construction delay correspondence",
            "billing dispute",
            "project schedule",
            "material cost increase",
            "subcontractor payment",
            "architectural drawings revision"
        ]
        
        # Warm-up search
        _ = search.search(queries[0])
        
        # Test basic search performance
        print("\nSemantic Search Performance (Basic):")
        start_time = time.time()
        for query in queries:
            results = search.search(query, limit=10)
            assert len(results) <= 10
        basic_search_time = time.time() - start_time
        
        print(f"Executed {len(queries)} searches in {basic_search_time:.4f} seconds")
        print(f"Average time per search: {(basic_search_time / len(queries)) * 1000:.4f} ms")
        
        # Test filtered search performance
        print("\nSemantic Search Performance (Filtered):")
        filters = [
            {"doc_type": "invoice"},
            {"doc_type": "change_order"},
            {"party": "contractor"},
            {"doc_type": "payment_app", "party": "contractor"},
            {"doc_type": "contract", "party": "owner"}
        ]
        
        start_time = time.time()
        for i, query in enumerate(queries[:5]):
            filter_dict = filters[i % len(filters)]
            results = search.search(query, filters=filter_dict, limit=10)
            assert len(results) <= 10
        filtered_search_time = time.time() - start_time
        
        print(f"Executed {len(queries[:5])} filtered searches in {filtered_search_time:.4f} seconds")
        print(f"Average time per filtered search: {(filtered_search_time / len(queries[:5])) * 1000:.4f} ms")
        
        # Test hybrid search performance
        print("\nSemantic Search Performance (Hybrid):")
        
        # Mock keyword search function
        with mock.patch('cdas.ai.semantic_search.search.keyword_search') as mock_keyword_search:
            # Return some mock results
            mock_keyword_search.return_value = [
                {"doc_id": f"doc{i}", "score": 0.9 - (i * 0.1), "text": f"Mock result {i}"} 
                for i in range(5)
            ]
            
            start_time = time.time()
            for query in queries:
                results = search.hybrid_search(query, limit=10)
                assert len(results) <= 10
            hybrid_search_time = time.time() - start_time
            
            print(f"Executed {len(queries)} hybrid searches in {hybrid_search_time:.4f} seconds")
            print(f"Average time per hybrid search: {(hybrid_search_time / len(queries)) * 1000:.4f} ms")


@pytest.mark.benchmark
def test_embedding_batch_performance(test_session):
    """Benchmark the performance of batch embedding operations."""
    # Create embedding manager in mock mode for benchmarking
    config = {"mock_mode": True}
    embedding_manager = EmbeddingManager(test_session, config)
    
    # Generate test documents of various sizes
    texts = []
    for i in range(10):
        # Small texts (100-500 chars)
        small_text = ''.join(random.choice(string.ascii_letters + ' ') for _ in range(random.randint(100, 500)))
        texts.append(small_text)
    
    for i in range(10):
        # Medium texts (1000-3000 chars)
        medium_text = ''.join(random.choice(string.ascii_letters + ' ') for _ in range(random.randint(1000, 3000)))
        texts.append(medium_text)
    
    for i in range(5):
        # Large texts (5000-10000 chars)
        large_text = ''.join(random.choice(string.ascii_letters + ' ') for _ in range(random.randint(5000, 10000)))
        texts.append(large_text)
    
    # Test individual embedding performance
    print("\nIndividual Embedding Performance:")
    start_time = time.time()
    embeddings = []
    for text in texts:
        embedding = embedding_manager.generate_embeddings(text)
        embeddings.append(embedding)
    individual_time = time.time() - start_time
    
    print(f"Generated {len(texts)} individual embeddings in {individual_time:.4f} seconds")
    print(f"Average time per embedding: {(individual_time / len(texts)) * 1000:.4f} ms")
    
    # Test batch embedding performance
    print("\nBatch Embedding Performance:")
    start_time = time.time()
    batch_embeddings = embedding_manager.generate_embeddings_batch(texts)
    batch_time = time.time() - start_time
    
    print(f"Generated {len(texts)} embeddings in batch in {batch_time:.4f} seconds")
    print(f"Average time per embedding: {(batch_time / len(texts)) * 1000:.4f} ms")
    print(f"Speedup from batching: {individual_time / batch_time:.2f}x")
    
    # Verify batch results match individual results
    for i in range(len(texts)):
        assert len(embeddings[i]) == len(batch_embeddings[i])
        assert embeddings[i] == batch_embeddings[i]
    
    # Test with different batch sizes
    batch_sizes = [1, 5, 10, 20, len(texts)]
    
    print("\nPerformance with Different Batch Sizes:")
    for batch_size in batch_sizes:
        # Process all texts in batches of the specified size
        start_time = time.time()
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_result = embedding_manager.generate_embeddings_batch(batch)
            results.extend(batch_result)
        
        batch_n_time = time.time() - start_time
        
        print(f"Batch size {batch_size}: {batch_n_time:.4f} seconds "
              f"({(batch_n_time / len(texts)) * 1000:.4f} ms per embedding)")
    
    # Test batching with real API client (mocked)
    with mock.patch('cdas.ai.embeddings.OpenAI') as mock_openai:
        # Setup mock client
        mock_client = mock.Mock()
        mock_openai.return_value = mock_client
        
        # Set up mock embedding response
        mock_response = mock.Mock()
        mock_data = []
        
        for i in range(len(texts)):
            mock_item = mock.Mock()
            mock_item.embedding = [0.1, 0.2, 0.3] * 512
            mock_data.append(mock_item)
        
        mock_response.data = mock_data
        mock_client.embeddings.create.return_value = mock_response
        
        # Create embedding manager with mocked client
        real_embedding_manager = EmbeddingManager(test_session, {"mock_mode": False})
        
        # Test batch performance with mocked API
        start_time = time.time()
        _ = real_embedding_manager.generate_embeddings_batch(texts)
        mocked_api_time = time.time() - start_time
        
        print(f"\nMocked API batch call: {mocked_api_time:.4f} seconds "
              f"({(mocked_api_time / len(texts)) * 1000:.4f} ms per embedding)")
        
        # Verify API was called with batched input
        mock_client.embeddings.create.assert_called_once()
        call_args = mock_client.embeddings.create.call_args[1]
        assert isinstance(call_args['input'], list)
        assert len(call_args['input']) == len(texts)


if __name__ == "__main__":
    pytest.main(["-v", __file__])