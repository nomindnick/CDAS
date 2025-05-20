"""Tests for semantic search functionality."""

import unittest
from unittest import mock
import numpy as np

from cdas.ai.semantic_search.search import semantic_search, batch_embed_documents, semantic_query
from cdas.ai.semantic_search.vectorizer import Vectorizer
from cdas.ai.semantic_search.index import VectorIndex


class TestSemanticSearch(unittest.TestCase):
    """Tests for semantic search functionality."""
    
    def setUp(self):
        """Set up test case."""
        self.db_session = mock.Mock()
        self.embedding_manager = mock.Mock()
        
        # Mock embedding generation
        self.embedding_manager.generate_embeddings.return_value = [0.1] * 1536
        
        # Mock db session execute
        self.execute_mock = mock.Mock()
        self.fetchall_mock = mock.Mock()
        self.fetchall_mock.fetchall.return_value = [
            (
                'page_1', 'doc_1', 1, 'Test content 1', 'Test Document 1', 
                'contract', 'contractor', mock.Mock(), 'active', 0.9
            ),
            (
                'page_2', 'doc_2', 1, 'Test content 2', 'Test Document 2', 
                'invoice', 'contractor', mock.Mock(), 'active', 0.8
            ),
        ]
        self.execute_mock.return_value = self.fetchall_mock
        self.db_session.execute.return_value = self.execute_mock
    
    def test_semantic_search_basic(self):
        """Test basic semantic search functionality."""
        # Mock the database session to return search results
        self.db_session.execute = mock.Mock()
        self.db_session.execute.return_value.fetchall.return_value = [
            (
                'page_1', 'doc_1', 1, 'Test content 1', 'Test Document 1', 
                'contract', 'contractor', mock.Mock(), 'active', 0.9
            ),
            (
                'page_2', 'doc_2', 1, 'Test content 2', 'Test Document 2', 
                'invoice', 'contractor', mock.Mock(), 'active', 0.8
            ),
        ]
        
        results = semantic_search(self.db_session, self.embedding_manager, 'test query')
        
        # Check that embeddings were generated
        self.embedding_manager.generate_embeddings.assert_called_once_with('test query')
        
        # Check that database was queried
        self.db_session.execute.assert_called_once()
        
        # Check search results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['doc_id'], 'doc_1')
        self.assertEqual(results[0]['similarity'], 0.9)
        self.assertEqual(results[0]['match_score'], 90)
        self.assertEqual(results[1]['doc_id'], 'doc_2')
        self.assertEqual(results[1]['similarity'], 0.8)
        self.assertEqual(results[1]['match_score'], 80)
    
    def test_semantic_search_with_filters(self):
        """Test semantic search with filters."""
        results = semantic_search(
            self.db_session, 
            self.embedding_manager, 
            'test query',
            doc_type='contract',
            party='contractor',
            min_date='2023-01-01',
            max_date='2023-12-31',
            threshold=0.5,
            metadata_filters={
                'status': 'active',
                'amount_min': 1000
            }
        )
        
        # Check that database was queried with filters
        self.db_session.execute.assert_called_once()
        call_args = self.db_session.execute.call_args[0]
        
        # Check that query contains filter conditions
        query = call_args[0]
        params = call_args[1]
        
        self.assertIn("d.doc_type = %s", query)
        self.assertIn("d.party = %s", query)
        self.assertIn("d.date >= %s", query)
        self.assertIn("d.date <= %s", query)
        self.assertIn("d.status = %s", query)
        self.assertIn("SELECT DISTINCT doc_id FROM line_items WHERE amount >= %s", query)
        
        # Check parameter values
        self.assertIn('contract', params)
        self.assertIn('contractor', params)
        self.assertIn('2023-01-01', params)
        self.assertIn('2023-12-31', params)
        self.assertIn('active', params)
        self.assertIn(1000, params)
    
    def test_batch_embed_documents(self):
        """Test batch embedding of documents."""
        # Mock the database session for document lookup
        self.db_session.execute = mock.Mock()
        self.db_session.execute.return_value.fetchall.return_value = [
            ('doc_1',),
            ('doc_2',),
        ]
        
        # Mock embedding_manager.embed_document
        self.embedding_manager.embed_document = mock.Mock(return_value=True)
        
        result = batch_embed_documents(
            self.db_session,
            self.embedding_manager,
            doc_type='contract',
            limit=10
        )
        
        # Check that database was queried
        self.db_session.execute.assert_called()
        
        # Check that embedding was created for each document
        self.assertEqual(self.embedding_manager.embed_document.call_count, 2)
        
        # Check result
        self.assertEqual(result['total_documents'], 2)
        self.assertEqual(result['embedded_documents'], 2)
        self.assertEqual(result['failed_documents'], 0)
        self.assertEqual(len(result['embedded_doc_ids']), 2)
        self.assertEqual(len(result['failed_doc_ids']), 0)
    
    def test_batch_embed_with_vectorizer(self):
        """Test batch embedding using the vectorizer."""
        # Mock document lookup
        self.db_session.execute = mock.Mock()
        
        # First call - get document IDs
        doc_ids_mock = mock.Mock()
        doc_ids_mock.fetchall.return_value = [('doc_1',)]
        
        # Second call - get document content and metadata
        doc_info_mock = mock.Mock()
        doc_info_mock.fetchone.return_value = (
            'doc_1', 'Test Document', 'contract', 'contractor', 
            mock.Mock(), 'active', 'Test content'
        )
        
        # Set up multiple return values for execute
        self.db_session.execute.side_effect = [
            doc_ids_mock,
            doc_info_mock,
            None  # For the insert query
        ]
        
        # Mock Vectorizer
        vectorizer_mock = mock.Mock()
        vectorizer_mock.vectorize_document.return_value = [
            {
                'text': 'Test chunk',
                'embedding': [0.1] * 1536
            }
        ]
        
        # Mock the import
        with mock.patch('cdas.ai.semantic_search.search.Vectorizer', return_value=vectorizer_mock):
            result = batch_embed_documents(
                self.db_session,
                self.embedding_manager,
                doc_type='contract',
                use_vectorizer=True
            )
        
        # Check that vectorizer was used
        vectorizer_mock.vectorize_document.assert_called_once()
        
        # Check that database was updated with chunks
        self.db_session.execute.assert_called()
        
        # Check result
        self.assertEqual(result['total_documents'], 1)
        self.assertEqual(result['embedded_documents'], 1)
        self.assertEqual(result['failed_documents'], 0)
    
    def test_semantic_query_hybrid(self):
        """Test hybrid semantic and keyword search."""
        # Mock semantic search results
        semantic_results = [
            {
                'page_id': 'page_1',
                'doc_id': 'doc_1',
                'page_number': 1,
                'document': {
                    'title': 'Test Document 1',
                    'doc_type': 'contract',
                    'party': 'contractor',
                    'date': '2023-01-01',
                    'status': 'active'
                },
                'context': 'Test content 1',
                'similarity': 0.9,
                'match_score': 90
            }
        ]
        
        # Mock keyword search results
        self.db_session.execute = mock.Mock()
        self.db_session.execute.return_value.fetchall.return_value = [
            (
                'page_3', 'doc_3', 1, 'Test content 3', 'Test Document 3', 
                'invoice', 'contractor', mock.Mock(), 'active', 0.7
            )
        ]
        
        # Mock semantic_search function
        with mock.patch('cdas.ai.semantic_search.search.semantic_search', return_value=semantic_results):
            results = semantic_query(
                self.db_session,
                self.embedding_manager,
                'test query',
                filters={
                    'doc_type': 'contract',
                    'party': 'contractor'
                },
                use_hybrid=True
            )
        
        # Check that both semantic and keyword results are included
        self.assertEqual(len(results), 2)
        
        # Check that results are sorted by match score
        self.assertEqual(results[0]['match_type'], 'semantic')
        self.assertEqual(results[0]['match_score'], 90)
        self.assertEqual(results[1]['match_type'], 'keyword')
        
        # Check that duplicate results are eliminated
        page_ids = [r['page_id'] for r in results]
        self.assertEqual(len(page_ids), len(set(page_ids)))


class TestVectorizer(unittest.TestCase):
    """Tests for the Vectorizer class."""
    
    def setUp(self):
        """Set up test case."""
        self.embedding_manager = mock.Mock()
        self.embedding_manager.generate_embeddings.return_value = [0.1] * 1536
        self.vectorizer = Vectorizer(self.embedding_manager)
    
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.vectorizer.embedding_manager, self.embedding_manager)
        self.assertEqual(self.vectorizer.chunk_size, 1000)
        self.assertEqual(self.vectorizer.chunk_overlap, 200)
        self.assertTrue(self.vectorizer.preserve_structure)
    
    def test_vectorize_text(self):
        """Test vectorizing text."""
        text = "This is a test document. " * 50  # Make it long enough to chunk
        metadata = {'doc_id': 'doc_1', 'doc_type': 'contract'}
        
        result = self.vectorizer.vectorize_text(text, metadata)
        
        # Check that text was chunked and embeddings were generated
        self.assertGreaterEqual(len(result), 1)
        self.embedding_manager.generate_embeddings.assert_called()
        
        # Check result structure
        for item in result:
            self.assertIn('text', item)
            self.assertIn('embedding', item)
            self.assertIn('metadata', item)
            self.assertEqual(item['metadata']['doc_id'], 'doc_1')
            self.assertEqual(item['metadata']['doc_type'], 'contract')
            self.assertIn('chunk_index', item['metadata'])
            self.assertIn('total_chunks', item['metadata'])
    
    def test_vectorize_document(self):
        """Test vectorizing a document."""
        document = {
            'doc_id': 'doc_1',
            'title': 'Test Document',
            'doc_type': 'contract',
            'party': 'contractor',
            'date': '2023-01-01',
            'status': 'active',
            'content': "This is a test document. " * 50
        }
        
        result = self.vectorizer.vectorize_document(document)
        
        # Check that document was vectorized
        self.assertGreaterEqual(len(result), 1)
        self.embedding_manager.generate_embeddings.assert_called()
        
        # Check result structure
        for item in result:
            self.assertIn('text', item)
            self.assertIn('embedding', item)
            self.assertIn('metadata', item)
            self.assertEqual(item['metadata']['doc_id'], 'doc_1')
            self.assertEqual(item['metadata']['doc_type'], 'contract')
    
    def test_special_handler_invoice(self):
        """Test document type-specific handler for invoices."""
        document = {
            'doc_id': 'inv_1',
            'title': 'Test Invoice',
            'doc_type': 'invoice',
            'party': 'contractor',
            'date': '2023-01-01',
            'status': 'active',
            'content': """
            Invoice #12345
            Date: 01/15/2023
            
            Item 1: $100.00
            Item 2: $200.00
            
            Total Amount: $300.00
            """
        }
        
        # Mock the vectorize_text method
        original_vectorize_text = self.vectorizer.vectorize_text
        self.vectorizer.vectorize_text = mock.Mock()
        
        # Call vectorize_document
        self.vectorizer.vectorize_document(document)
        
        # Check that invoice-specific handling was applied
        call_args = self.vectorizer.vectorize_text.call_args
        metadata = call_args[0][1]
        
        self.assertEqual(metadata['invoice_number'], '12345')
        self.assertEqual(metadata['total_amount'], '300.00')
        
        # Restore original method
        self.vectorizer.vectorize_text = original_vectorize_text


class TestVectorIndex(unittest.TestCase):
    """Tests for the VectorIndex class."""
    
    def setUp(self):
        """Set up test case."""
        self.index = VectorIndex(config={'index_name': 'test_index'})
    
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.index.index_name, 'test_index')
        self.assertEqual(self.index.dimensions, 1536)
        self.assertEqual(len(self.index.vectors), 0)
        self.assertEqual(len(self.index.metadata), 0)
        self.assertEqual(len(self.index.id_to_index), 0)
    
    def test_add_vector(self):
        """Test adding a vector to the index."""
        vector_id = 'vec_1'
        vector = [0.1] * 1536
        metadata = {'doc_id': 'doc_1', 'doc_type': 'contract'}
        
        result = self.index.add_vector(vector_id, vector, metadata)
        
        self.assertTrue(result)
        self.assertEqual(len(self.index.vectors), 1)
        self.assertEqual(len(self.index.metadata), 1)
        self.assertEqual(len(self.index.id_to_index), 1)
        self.assertEqual(self.index.total_vectors, 1)
        self.assertTrue(self.index.is_dirty)
    
    def test_batch_add_vectors(self):
        """Test adding multiple vectors to the index."""
        vector_data = [
            ('vec_1', [0.1] * 1536, {'doc_id': 'doc_1'}),
            ('vec_2', [0.2] * 1536, {'doc_id': 'doc_2'})
        ]
        
        result = self.index.batch_add_vectors(vector_data)
        
        self.assertEqual(result['added'], 2)
        self.assertEqual(result['updated'], 0)
        self.assertEqual(result['failed'], 0)
        self.assertTrue(result['success'])
        self.assertEqual(len(self.index.vectors), 2)
        self.assertEqual(len(self.index.metadata), 2)
        self.assertEqual(len(self.index.id_to_index), 2)
        self.assertEqual(self.index.total_vectors, 2)
        self.assertTrue(self.index.is_dirty)
    
    def test_remove_vector(self):
        """Test removing a vector from the index."""
        # Add vectors first
        self.index.add_vector('vec_1', [0.1] * 1536, {'doc_id': 'doc_1'})
        self.index.add_vector('vec_2', [0.2] * 1536, {'doc_id': 'doc_2'})
        
        result = self.index.remove_vector('vec_1')
        
        self.assertTrue(result)
        self.assertEqual(len(self.index.vectors), 1)
        self.assertEqual(len(self.index.metadata), 1)
        self.assertEqual(len(self.index.id_to_index), 1)
        self.assertEqual(self.index.total_vectors, 1)
        self.assertTrue('vec_1' not in self.index.id_to_index)
        self.assertTrue('vec_2' in self.index.id_to_index)
    
    def test_search(self):
        """Test searching for similar vectors."""
        # Add vectors
        with mock.patch('numpy.linalg.norm', return_value=1.0):
            with mock.patch('numpy.dot', return_value=np.array([0.9, 0.5])):
                self.index.add_vector('vec_1', [0.1] * 1536, {'doc_id': 'doc_1'})
                self.index.add_vector('vec_2', [0.2] * 1536, {'doc_id': 'doc_2'})
                
                results = self.index.search([0.3] * 1536, limit=2)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['vector_id'], 'vec_1')
        self.assertEqual(results[0]['similarity'], 0.9)
        self.assertEqual(results[0]['metadata']['doc_id'], 'doc_1')
        self.assertEqual(results[1]['vector_id'], 'vec_2')
        self.assertEqual(results[1]['similarity'], 0.5)
        self.assertEqual(results[1]['metadata']['doc_id'], 'doc_2')
    
    def test_filtered_search(self):
        """Test searching with metadata filters."""
        # Add vectors
        self.index.add_vector('vec_1', [0.1] * 1536, {
            'doc_id': 'doc_1', 
            'doc_type': 'contract',
            'party': 'contractor'
        })
        self.index.add_vector('vec_2', [0.2] * 1536, {
            'doc_id': 'doc_2', 
            'doc_type': 'invoice',
            'party': 'contractor'
        })
        
        with mock.patch('numpy.linalg.norm', return_value=1.0):
            with mock.patch('numpy.dot', return_value=np.array([0.9])):
                results = self.index.filtered_search(
                    [0.3] * 1536, 
                    metadata_filters={'doc_type': 'contract'}
                )
        
        # Check that only filtered results are returned
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['vector_id'], 'vec_1')
        self.assertEqual(results[0]['metadata']['doc_type'], 'contract')
    
    def test_save_and_load_index(self):
        """Test saving and loading the index."""
        # Add vectors
        self.index.add_vector('vec_1', [0.1] * 1536, {'doc_id': 'doc_1'})
        
        # Mock file operations
        with mock.patch('builtins.open', mock.mock_open()):
            with mock.patch('pickle.dump') as mock_dump:
                with mock.patch('os.replace') as mock_replace:
                    result = self.index.save_index()
        
        self.assertTrue(result)
        mock_dump.assert_called_once()
        mock_replace.assert_called_once()
        
        # Reset index
        self.index.clear_index()
        self.assertEqual(len(self.index.vectors), 0)
        
        # Load index
        with mock.patch('builtins.open', mock.mock_open()):
            with mock.patch('pickle.load') as mock_load:
                mock_load.return_value = {
                    'dimensions': 1536,
                    'vectors': [[0.1] * 1536],
                    'metadata': [{'doc_id': 'doc_1'}],
                    'id_to_index': {'vec_1': 0},
                    'total_vectors': 1,
                    'last_modified': 123456789
                }
                result = self.index.load_index()
        
        self.assertTrue(result)
        self.assertEqual(len(self.index.vectors), 1)
        self.assertEqual(len(self.index.metadata), 1)
        self.assertEqual(len(self.index.id_to_index), 1)
        self.assertEqual(self.index.total_vectors, 1)


if __name__ == '__main__':
    unittest.main()