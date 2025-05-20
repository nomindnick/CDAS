"""Tests for the LLM manager."""

import unittest
from unittest import mock

from cdas.ai.llm import LLMManager


class TestLLMManager(unittest.TestCase):
    """Tests for the LLMManager class."""
    
    def test_init_defaults(self):
        """Test initialization with defaults."""
        with mock.patch('cdas.ai.llm.OpenAI') as mock_openai:
            llm_manager = LLMManager()
            
            self.assertEqual(llm_manager.model, 'o4-mini')
            self.assertEqual(llm_manager.provider, 'openai')
            self.assertEqual(llm_manager.temperature, 0.0)
            self.assertEqual(llm_manager.reasoning_effort, 'medium')
            mock_openai.assert_called_once()
    
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = {
            'model': 'gpt-3.5-turbo',
            'provider': 'openai',
            'temperature': 0.5,
            'reasoning_effort': 'high',
            'api_key': 'test_key'
        }
        
        with mock.patch('cdas.ai.llm.OpenAI') as mock_openai:
            llm_manager = LLMManager(config)
            
            self.assertEqual(llm_manager.model, 'gpt-3.5-turbo')
            self.assertEqual(llm_manager.provider, 'openai')
            self.assertEqual(llm_manager.temperature, 0.5)
            self.assertEqual(llm_manager.reasoning_effort, 'high')
            mock_openai.assert_called_once_with(api_key='test_key')
    
    def test_generate_openai(self):
        """Test text generation with OpenAI."""
        mock_client = mock.Mock()
        mock_response = mock.Mock()
        mock_choice = mock.Mock()
        mock_message = mock.Mock()
        
        mock_message.content = 'Generated text'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        with mock.patch('cdas.ai.llm.OpenAI', return_value=mock_client):
            llm_manager = LLMManager()
            result = llm_manager.generate('Test prompt', 'System prompt')
            
            self.assertEqual(result, 'Generated text')
            # Check if parameters were passed correctly
            call_args = mock_client.chat.completions.create.call_args[1]
            self.assertEqual(call_args['model'], 'o4-mini')
            self.assertEqual(call_args['messages'], [
                {'role': 'system', 'content': 'System prompt'},
                {'role': 'user', 'content': 'Test prompt'}
            ])
            self.assertEqual(call_args['temperature'], 0.0)
            self.assertEqual(call_args['reasoning_effort'], 'medium')


if __name__ == '__main__':
    unittest.main()
