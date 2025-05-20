"""
Tests for OpenAI o4-mini model integration with LLMManager.

These tests focus on the LLMManager's ability to work with
OpenAI's o4-mini model and its reasoning_effort parameter.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from cdas.ai.llm import LLMManager


class TestO4MiniModel:
    """Test the LLMManager with o4-mini model."""

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        with patch('cdas.ai.llm.OpenAI') as mock_openai:
            # Create a manager with default config
            manager = LLMManager()
            
            # Verify defaults
            assert manager.provider == 'openai'
            assert manager.model == 'o4-mini'
            assert manager.reasoning_effort == 'medium'
            
            # Verify OpenAI client was initialized
            mock_openai.assert_called_once()

    def test_o4_mini_reasoning_effort_parameter(self):
        """Test o4-mini model uses reasoning_effort parameter."""
        with patch('cdas.ai.llm.OpenAI') as mock_openai:
            # Setup mock
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Test response"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            
            # Create manager
            manager = LLMManager({
                'model': 'o4-mini',
                'reasoning_effort': 'high'
            })
            
            # Generate text
            response = manager.generate("Test prompt")
            
            # Verify reasoning_effort was included
            call_args = mock_client.chat.completions.create.call_args[1]
            assert call_args['reasoning_effort'] == 'high'
            assert call_args['model'] == 'o4-mini'
            assert response == "Test response"

    def test_generate_with_tools_includes_reasoning_effort(self):
        """Test generate_with_tools includes reasoning_effort for o4 models."""
        with patch('cdas.ai.llm.OpenAI') as mock_openai:
            # Setup mock
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Test response"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            
            # Create manager
            manager = LLMManager({
                'model': 'o4-mini',
                'reasoning_effort': 'high'
            })
            
            # Define a simple tool
            tools = [{
                "type": "function",
                "function": {
                    "name": "test_function",
                    "description": "A test function",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param1": {
                                "type": "string",
                                "description": "A test parameter"
                            }
                        }
                    }
                }
            }]
            
            # Generate text with tools
            response = manager.generate_with_tools("Test prompt with tools", tools)
            
            # Verify reasoning_effort was included
            call_args = mock_client.chat.completions.create.call_args[1]
            assert call_args['reasoning_effort'] == 'high'
            assert call_args['model'] == 'o4-mini'
            assert 'tools' in call_args
            assert call_args['tools'] == tools

    def test_non_o4_model_skips_reasoning_effort(self):
        """Test non-o4 models don't include reasoning_effort parameter."""
        with patch('cdas.ai.llm.OpenAI') as mock_openai:
            # Setup mock
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Test response"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            
            # Create manager with GPT-4 model
            manager = LLMManager({
                'model': 'gpt-4',
                'reasoning_effort': 'high'  # This should be ignored
            })
            
            # Generate text
            response = manager.generate("Test prompt")
            
            # Verify reasoning_effort was NOT included
            call_args = mock_client.chat.completions.create.call_args[1]
            assert 'reasoning_effort' not in call_args
            assert call_args['model'] == 'gpt-4'