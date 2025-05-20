"""Tests for the embeddings manager."""

import unittest
from unittest import mock

from cdas.ai.embeddings import EmbeddingManager


class TestEmbeddingManager(unittest.TestCase):
    """Tests for the EmbeddingManager class."""
    
    def test_init_defaults(self):
        """Test initialization with defaults."""
        db_session = mock.Mock()
        
        with mock.patch('cdas.ai.embeddings.OpenAI') as mock_openai:
            embedding_manager = EmbeddingManager(db_session)
            
            self.assertEqual(embedding_manager.embedding_model, 'text-embedding-3-small')
            mock_openai.assert_called_once()
    
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        db_session = mock.Mock()
        config = {
            'embedding_model': 'text-embedding-3-large',
            'api_key': 'test_key'
        }
        
        with mock.patch('cdas.ai.embeddings.OpenAI') as mock_openai:
            embedding_manager = EmbeddingManager(db_session, config)
            
            self.assertEqual(embedding_manager.embedding_model, 'text-embedding-3-large')
            mock_openai.assert_called_once_with(api_key='test_key')
    
    def test_generate_embeddings(self):
        """Test embedding generation."""
        db_session = mock.Mock()
        mock_client = mock.Mock()
        mock_response = mock.Mock()
        mock_data = mock.Mock()
        
        mock_data.embedding = [0.1, 0.2, 0.3]
        mock_response.data = [mock_data]
        mock_client.embeddings.create.return_value = mock_response
        
        with mock.patch('cdas.ai.embeddings.OpenAI', return_value=mock_client):
            embedding_manager = EmbeddingManager(db_session)
            result = embedding_manager.generate_embeddings('Test text')
            
            self.assertEqual(result, [0.1, 0.2, 0.3])
            mock_client.embeddings.create.assert_called_once_with(
                model='text-embedding-3-small',
                input='Test text'
            )


if __name__ == '__main__':
    unittest.main()
