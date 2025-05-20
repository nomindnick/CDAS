"""
Configuration management for the Construction Document Analysis System.

This module provides functions for loading and managing configuration settings,
including environment-specific configurations.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Default configuration
_DEFAULT_CONFIG = {
    "database": {
        "db_type": "sqlite",
        "db_path": str(Path.home() / ".cdas" / "cdas.db"),
        "echo": False
    },
    "document_processor": {
        "pdf_processor": "pdfplumber",
        "ocr_engine": "tesseract",
        "handwriting_recognition": True,
        "extraction_confidence_threshold": 0.5
    },
    "analysis": {
        "amount_matching_tolerance": 0.01,
        "pattern_confidence_threshold": 0.7,
        "anomaly_detection_threshold": 0.8
    },
    "ai": {
        "llm_model": "o4-mini",
        "embedding_model": "text-embedding-3-small",
        "reasoning_effort": "medium"
    },
    "reporting": {
        "default_format": "pdf",
        "evidence_citation_format": "{doc_type} {doc_number}, page {page_number}",
        "templates_dir": "cdas/reporting/templates"
    },
    "logging": {
        "level": "INFO",
        "file": str(Path.home() / ".cdas" / "logs" / "cdas.log"),
        "max_size": 10485760,  # 10 MB
        "backup_count": 5
    }
}

# Global configuration dictionary
_CONFIG = None


def _load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load config file {config_path}: {e}")
        return {}


def get_config() -> Dict[str, Any]:
    """Get the current configuration.
    
    Returns:
        Configuration dictionary
    """
    global _CONFIG
    
    if _CONFIG is None:
        # Start with default configuration
        _CONFIG = _DEFAULT_CONFIG.copy()
        
        # Look for configuration files in standard locations
        config_paths = [
            os.path.join(os.getcwd(), "cdas.json"),
            os.path.join(str(Path.home()), ".cdas", "config.json"),
            os.environ.get("CDAS_CONFIG", "")
        ]
        
        # Override with configuration from files
        for path in config_paths:
            if path and os.path.exists(path):
                file_config = _load_config_file(path)
                _merge_configs(_CONFIG, file_config)
        
        # Override with environment variables
        _apply_env_overrides(_CONFIG)
    
    return _CONFIG


def _merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> None:
    """Merge override configuration into base configuration.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
    """
    for key, value in override_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            _merge_configs(base_config[key], value)
        else:
            base_config[key] = value


def _apply_env_overrides(config: Dict[str, Any], prefix: str = "CDAS_") -> None:
    """Apply environment variable overrides to configuration.
    
    Args:
        config: Configuration dictionary
        prefix: Environment variable prefix
    """
    for env_var, env_value in os.environ.items():
        if env_var.startswith(prefix):
            # Remove prefix and split by double underscore to get nested keys
            keys = env_var[len(prefix):].lower().split("__")
            
            # Navigate to the correct nested dictionary
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                elif not isinstance(current[key], dict):
                    current[key] = {}
                current = current[key]
            
            # Set the value, converting to appropriate type
            try:
                # Try to convert to int or float if possible
                if env_value.isdigit():
                    current[keys[-1]] = int(env_value)
                elif env_value.replace(".", "", 1).isdigit() and env_value.count(".") == 1:
                    current[keys[-1]] = float(env_value)
                elif env_value.lower() == "true":
                    current[keys[-1]] = True
                elif env_value.lower() == "false":
                    current[keys[-1]] = False
                else:
                    current[keys[-1]] = env_value
            except (ValueError, TypeError):
                current[keys[-1]] = env_value


def set_config(new_config: Dict[str, Any]) -> None:
    """Set a new configuration.
    
    Args:
        new_config: New configuration dictionary
    """
    global _CONFIG
    _CONFIG = _DEFAULT_CONFIG.copy()
    _merge_configs(_CONFIG, new_config)


def reset_config() -> None:
    """Reset configuration to default."""
    global _CONFIG
    _CONFIG = None