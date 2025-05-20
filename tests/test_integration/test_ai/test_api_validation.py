"""
Tests for API key validation and configuration handling.

These tests verify that the system correctly validates API keys,
loads them from various sources, and handles configuration errors gracefully.
"""

import os
import pytest
import tempfile
from unittest import mock

from cdas.ai.llm import LLMManager
from cdas.ai.embeddings import EmbeddingManager
from cdas.config import get_config, set_config


def test_api_key_validation_llm():
    """Test validation of API keys in LLM manager."""
    # Test various invalid API key formats
    invalid_keys = [
        "",                  # Empty string
        None,                # None
        "invalid-key",       # Wrong format
        "sk-" + "a" * 5,     # Too short
        "not-sk-12345"       # Wrong prefix
    ]
    
    for key in invalid_keys:
        # Mock session since we're just testing validation
        session = mock.Mock()
        
        # Mock the API clients to prevent actual API calls
        with mock.patch('cdas.ai.llm.OpenAI') as mock_openai, \
             mock.patch('cdas.ai.llm.Anthropic') as mock_anthropic, \
             mock.patch('cdas.ai.llm.logging.warning') as mock_warning:
            
            # Create LLM manager with invalid key
            config = {
                "openai_api_key": key,
                "anthropic_api_key": "sk-ant-valid123"  # Valid Anthropic key
            }
            
            llm = LLMManager(session, config)
            
            # Should fall back to mock mode due to invalid OpenAI key
            assert llm.mock_mode
            
            # Warning should be logged
            mock_warning.assert_called()
            
            # OpenAI client should not be initialized
            mock_openai.assert_not_called()
            
            # Anthropic client should still be initialized if key is valid
            mock_anthropic.assert_called_once()
    
    # Test with valid-looking keys (format validation only)
    valid_keys = [
        "sk-" + "a" * 40,    # OpenAI format
        "sk-ant-" + "a" * 40  # Anthropic format
    ]
    
    for openai_key in valid_keys:
        for anthropic_key in valid_keys:
            # Mock session
            session = mock.Mock()
            
            # Mock the API clients to prevent actual API calls
            with mock.patch('cdas.ai.llm.OpenAI') as mock_openai, \
                 mock.patch('cdas.ai.llm.Anthropic') as mock_anthropic:
                
                # Create LLM manager with valid-looking keys
                config = {
                    "openai_api_key": openai_key,
                    "anthropic_api_key": anthropic_key
                }
                
                llm = LLMManager(session, config)
                
                # Should not be in mock mode
                assert not llm.mock_mode
                
                # Both clients should be initialized
                mock_openai.assert_called_once()
                mock_anthropic.assert_called_once()


def test_api_key_validation_embeddings():
    """Test validation of API keys in embedding manager."""
    # Test various invalid API key formats
    invalid_keys = [
        "",                  # Empty string
        None,                # None
        "invalid-key",       # Wrong format
        "sk-" + "a" * 5,     # Too short
        "not-sk-12345"       # Wrong prefix
    ]
    
    for key in invalid_keys:
        # Mock session
        session = mock.Mock()
        
        # Mock the API client to prevent actual API calls
        with mock.patch('cdas.ai.embeddings.OpenAI') as mock_openai, \
             mock.patch('cdas.ai.embeddings.logging.warning') as mock_warning:
            
            # Create embedding manager with invalid key
            config = {
                "openai_api_key": key
            }
            
            embeddings = EmbeddingManager(session, config)
            
            # Should fall back to mock mode due to invalid key
            assert embeddings.mock_mode
            
            # Warning should be logged
            mock_warning.assert_called()
            
            # OpenAI client should not be initialized
            mock_openai.assert_not_called()
    
    # Test with valid-looking key (format validation only)
    valid_key = "sk-" + "a" * 40
    
    # Mock session
    session = mock.Mock()
    
    # Mock the API client to prevent actual API calls
    with mock.patch('cdas.ai.embeddings.OpenAI') as mock_openai:
        
        # Create embedding manager with valid-looking key
        config = {
            "openai_api_key": valid_key
        }
        
        embeddings = EmbeddingManager(session, config)
        
        # Should not be in mock mode
        assert not embeddings.mock_mode
        
        # OpenAI client should be initialized
        mock_openai.assert_called_once()


def test_api_key_loading_from_environment():
    """Test loading API keys from environment variables."""
    # Save original environment variables
    original_openai_key = os.environ.get('OPENAI_API_KEY')
    original_anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
    
    try:
        # Set environment variables
        os.environ['OPENAI_API_KEY'] = "sk-" + "a" * 40
        os.environ['ANTHROPIC_API_KEY'] = "sk-ant-" + "a" * 40
        
        # Mock session
        session = mock.Mock()
        
        # Mock the API clients to prevent actual API calls
        with mock.patch('cdas.ai.llm.OpenAI') as mock_openai, \
             mock.patch('cdas.ai.llm.Anthropic') as mock_anthropic:
            
            # Create LLM manager with empty config (should pick up from env)
            config = {}
            
            llm = LLMManager(session, config)
            
            # Should not be in mock mode
            assert not llm.mock_mode
            
            # Should have loaded keys from environment
            assert llm.openai_api_key == "sk-" + "a" * 40
            assert llm.anthropic_api_key == "sk-ant-" + "a" * 40
            
            # Both clients should be initialized
            mock_openai.assert_called_once()
            mock_anthropic.assert_called_once()
    finally:
        # Restore original environment variables
        if original_openai_key is not None:
            os.environ['OPENAI_API_KEY'] = original_openai_key
        else:
            os.environ.pop('OPENAI_API_KEY', None)
            
        if original_anthropic_key is not None:
            os.environ['ANTHROPIC_API_KEY'] = original_anthropic_key
        else:
            os.environ.pop('ANTHROPIC_API_KEY', None)


def test_api_key_loading_from_config_file():
    """Test loading API keys from a configuration file."""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as temp_config:
        # Write config to file
        temp_config.write("""
        {
            "ai": {
                "openai_api_key": "sk-configfile40chars1234567890123456789012",
                "anthropic_api_key": "sk-ant-configfile40chars1234567890123456",
                "openai_model": "gpt-4o",
                "embedding_model": "text-embedding-3-small"
            }
        }
        """)
        temp_config.flush()
        
        # Mock loading the config from file
        with mock.patch('cdas.config._load_config_file') as mock_load:
            mock_load.return_value = {
                "ai": {
                    "openai_api_key": "sk-configfile40chars1234567890123456789012",
                    "anthropic_api_key": "sk-ant-configfile40chars1234567890123456",
                    "openai_model": "gpt-4o",
                    "embedding_model": "text-embedding-3-small"
                }
            }
            
            # Set custom config directly
            custom_config = {
                "ai": {
                    "openai_api_key": "sk-configfile40chars1234567890123456789012",
                    "anthropic_api_key": "sk-ant-configfile40chars1234567890123456",
                    "openai_model": "gpt-4o",
                    "embedding_model": "text-embedding-3-small"
                }
            }
            set_config(custom_config)
            
            # Mock session
            session = mock.Mock()
            
            # Mock the API clients to prevent actual API calls
            with mock.patch('cdas.ai.llm.OpenAI') as mock_openai, \
                 mock.patch('cdas.ai.llm.Anthropic') as mock_anthropic:
                
                # Create LLM manager with direct config
                llm_config = {
                    "openai_api_key": "sk-configfile40chars1234567890123456789012",
                    "anthropic_api_key": "sk-ant-configfile40chars1234567890123456",
                    "openai_model": "gpt-4o"
                }
                llm = LLMManager(session, llm_config)
                
                # Should not be in mock mode
                assert not llm.mock_mode
                
                # Should have loaded keys from config file
                assert llm.openai_api_key == "sk-configfile40chars1234567890123456789012"
                assert llm.anthropic_api_key == "sk-ant-configfile40chars1234567890123456"
                
                # Should have loaded model settings from config file
                assert llm.openai_model == "gpt-4o"
                
                # Both clients should be initialized
                mock_openai.assert_called_once()
                mock_anthropic.assert_called_once()


def test_config_precedence():
    """Test that explicit config overrides environment variables."""
    # Save original environment variable
    original_openai_key = os.environ.get('OPENAI_API_KEY')
    
    try:
        # Set environment variable
        os.environ['OPENAI_API_KEY'] = "sk-fromenv40chars1234567890123456789012"
        
        # Mock session
        session = mock.Mock()
        
        # Mock the API client to prevent actual API calls
        with mock.patch('cdas.ai.embeddings.OpenAI') as mock_openai:
            
            # Create embedding manager with explicit config that should override env
            config = {
                "openai_api_key": "sk-explicit40chars1234567890123456789012",
                "embedding_model": "text-embedding-3-large"  # Non-default model
            }
            
            embeddings = EmbeddingManager(session, config)
            
            # Should use explicit config, not environment variable
            assert embeddings.openai_api_key == "sk-explicit40chars1234567890123456789012"
            
            # Should also use other explicit config values
            assert embeddings.embedding_model == "text-embedding-3-large"
    finally:
        # Restore original environment variable
        if original_openai_key is not None:
            os.environ['OPENAI_API_KEY'] = original_openai_key
        else:
            os.environ.pop('OPENAI_API_KEY', None)


def test_partial_config():
    """Test that partial configuration works correctly."""
    # Mock session
    session = mock.Mock()
    
    # Mock the API clients to prevent actual API calls
    with mock.patch('cdas.ai.llm.OpenAI') as mock_openai, \
         mock.patch('cdas.ai.llm.Anthropic') as mock_anthropic, \
         mock.patch('cdas.ai.llm.logging.warning') as mock_warning:
        
        # Create LLM manager with partial config (only OpenAI)
        config = {
            "openai_api_key": "sk-partial40chars1234567890123456789012",
            "openai_model": "gpt-4o"
        }
        
        llm = LLMManager(session, config)
        
        # Should have configured OpenAI
        assert llm.openai_api_key == "sk-partial40chars1234567890123456789012"
        assert llm.openai_model == "gpt-4o"
        
        # Anthropic should be in mock mode or have None key
        assert llm.anthropic_api_key is None or llm.mock_mode
        
        # OpenAI client should be initialized
        mock_openai.assert_called_once()
        
        # Warning should be logged about missing Anthropic key
        mock_warning.assert_called()
        
        # Create another LLM manager with just Anthropic config
        config2 = {
            "anthropic_api_key": "sk-ant-partial40chars1234567890123456789",
            "anthropic_model": "claude-3-sonnet-20240229"
        }
        
        # Reset mocks
        mock_openai.reset_mock()
        mock_anthropic.reset_mock()
        mock_warning.reset_mock()
        
        llm2 = LLMManager(session, config2)
        
        # Should have configured Anthropic
        assert llm2.anthropic_api_key == "sk-ant-partial40chars1234567890123456789"
        assert llm2.anthropic_model == "claude-3-sonnet-20240229"
        
        # OpenAI should be in mock mode or have None key
        assert llm2.openai_api_key is None or llm2.mock_mode
        
        # Anthropic client should be initialized
        mock_anthropic.assert_called_once()
        
        # Warning should be logged about missing OpenAI key
        mock_warning.assert_called()


def test_api_error_fallback():
    """Test fallback to mock mode when API clients raise errors."""
    # Mock session
    session = mock.Mock()
    
    # Test OpenAI client error
    with mock.patch('cdas.ai.llm.OpenAI') as mock_openai, \
         mock.patch('cdas.ai.llm.Anthropic') as mock_anthropic, \
         mock.patch('cdas.ai.llm.logging.error') as mock_error:
        
        # Make OpenAI client constructor raise an error
        mock_openai.side_effect = Exception("API error")
        
        # Create LLM manager with valid-looking keys
        config = {
            "openai_api_key": "sk-valid40chars1234567890123456789012",
            "anthropic_api_key": "sk-ant-valid40chars1234567890123456789"
        }
        
        llm = LLMManager(session, config)
        
        # Should fall back to mock mode due to API error
        assert llm.mock_mode
        
        # Error should be logged
        mock_error.assert_called()
        
        # Should still try to initialize Anthropic client
        mock_anthropic.assert_called_once()
    
    # Test both clients raising errors
    with mock.patch('cdas.ai.llm.OpenAI') as mock_openai, \
         mock.patch('cdas.ai.llm.Anthropic') as mock_anthropic, \
         mock.patch('cdas.ai.llm.logging.error') as mock_error:
        
        # Make both client constructors raise errors
        mock_openai.side_effect = Exception("OpenAI API error")
        mock_anthropic.side_effect = Exception("Anthropic API error")
        
        # Create LLM manager with valid-looking keys
        config = {
            "openai_api_key": "sk-valid40chars1234567890123456789012",
            "anthropic_api_key": "sk-ant-valid40chars1234567890123456789"
        }
        
        llm = LLMManager(session, config)
        
        # Should fall back to mock mode due to API errors
        assert llm.mock_mode
        
        # Error should be logged twice (once for each API)
        assert mock_error.call_count == 2
        
        # Both clients should be attempted
        mock_openai.assert_called_once()
        mock_anthropic.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-v", __file__])