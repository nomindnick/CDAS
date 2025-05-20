"""
Configuration utilities for the Construction Document Analysis System.

This module provides utilities for managing configuration across different
components of the system, with support for hierarchical configurations,
environment variable integration, and type validation.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Type, TypeVar, Union, cast, get_type_hints
from pathlib import Path
import json
import yaml
from dataclasses import dataclass, is_dataclass, asdict

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass

def load_config_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a file (JSON or YAML).
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigError: If file cannot be loaded
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
        
    if not file_path.exists():
        raise ConfigError(f"Configuration file not found: {file_path}")
        
    try:
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() in ['.yml', '.yaml']:
                try:
                    import yaml
                    return yaml.safe_load(f)
                except ImportError:
                    raise ConfigError("PyYAML package not installed. Install with: pip install pyyaml")
            elif file_path.suffix.lower() == '.json':
                import json
                return json.load(f)
            else:
                raise ConfigError(f"Unsupported config file format: {file_path.suffix}")
    except Exception as e:
        raise ConfigError(f"Error loading configuration: {str(e)}")

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration (takes precedence)
        
    Returns:
        Merged configuration dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        # If both values are dictionaries, merge them recursively
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
            
    return result

def get_env_config(prefix: str) -> Dict[str, Any]:
    """
    Get configuration from environment variables with a given prefix.
    
    Args:
        prefix: Environment variable prefix
        
    Returns:
        Configuration dictionary from environment variables
    """
    result = {}
    prefix_upper = prefix.upper() + "_"
    
    for key, value in os.environ.items():
        if key.startswith(prefix_upper):
            config_key = key[len(prefix_upper):].lower()
            
            # Handle nested config with double underscore
            if "__" in config_key:
                parts = config_key.split("__")
                current = result
                
                # Navigate to nested dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                    
                # Set the value
                current[parts[-1]] = parse_config_value(value)
            else:
                result[config_key] = parse_config_value(value)
                
    return result

def parse_config_value(value: str) -> Any:
    """
    Parse a configuration value from string to appropriate type.
    
    Args:
        value: String value to parse
        
    Returns:
        Parsed value
    """
    # Handle boolean values
    if value.lower() in ['true', 'yes', 'on', '1']:
        return True
    if value.lower() in ['false', 'no', 'off', '0']:
        return False
        
    # Handle None values
    if value.lower() in ['none', 'null']:
        return None
        
    # Handle numeric values
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        pass
        
    # Handle lists (comma-separated)
    if ',' in value:
        return [parse_config_value(v.strip()) for v in value.split(',')]
        
    # Default to string
    return value

def validate_config(config: Dict[str, Any], schema: Dict[str, Type]) -> Dict[str, Any]:
    """
    Validate configuration against a schema.
    
    Args:
        config: Configuration dictionary
        schema: Schema dictionary mapping keys to expected types
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ConfigError: If validation fails
    """
    validated = {}
    
    for key, expected_type in schema.items():
        if key not in config:
            if getattr(expected_type, '__origin__', None) is Union and type(None) in expected_type.__args__:
                # Optional field, skip
                continue
            else:
                # Required field
                raise ConfigError(f"Required configuration key missing: {key}")
                
        value = config[key]
        
        # Check type
        if value is not None:
            try:
                # Handle Union types
                if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
                    valid_types = [t for t in expected_type.__args__ if t is not type(None)]
                    if not any(isinstance(value, t) for t in valid_types):
                        raise ConfigError(f"Invalid type for {key}: expected one of {valid_types}, got {type(value)}")
                # Handle List types
                elif hasattr(expected_type, '__origin__') and expected_type.__origin__ is list:
                    if not isinstance(value, list):
                        raise ConfigError(f"Invalid type for {key}: expected list, got {type(value)}")
                # Handle Dict types
                elif hasattr(expected_type, '__origin__') and expected_type.__origin__ is dict:
                    if not isinstance(value, dict):
                        raise ConfigError(f"Invalid type for {key}: expected dict, got {type(value)}")
                # Handle simple types
                elif not isinstance(value, expected_type):
                    raise ConfigError(f"Invalid type for {key}: expected {expected_type}, got {type(value)}")
            except Exception as e:
                raise ConfigError(f"Error validating {key}: {str(e)}")
                
        validated[key] = value
        
    return validated

def load_config(config_path: Optional[Union[str, Path]] = None, 
               env_prefix: str = 'CDAS',
               default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration from multiple sources.
    
    Priority (highest to lowest):
    1. Environment variables
    2. Config file
    3. Default config
    
    Args:
        config_path: Path to configuration file
        env_prefix: Environment variable prefix
        default_config: Default configuration
        
    Returns:
        Merged configuration dictionary
    """
    # Start with default config
    config = default_config or {}
    
    # Load from file if provided
    if config_path:
        try:
            file_config = load_config_file(config_path)
            config = merge_configs(config, file_config)
        except ConfigError as e:
            logger.warning(f"Error loading config file: {str(e)}")
    
    # Override with environment variables
    env_config = get_env_config(env_prefix)
    if env_config:
        config = merge_configs(config, env_config)
        
    return config

def config_as_dataclass(config: Dict[str, Any], dataclass_type: Type[T]) -> T:
    """
    Convert a configuration dictionary to a dataclass instance.
    
    Args:
        config: Configuration dictionary
        dataclass_type: Dataclass type
        
    Returns:
        Dataclass instance
        
    Raises:
        ConfigError: If conversion fails
    """
    if not is_dataclass(dataclass_type):
        raise ConfigError(f"Type {dataclass_type} is not a dataclass")
        
    # Get type hints for validation
    type_hints = get_type_hints(dataclass_type)
    
    # Validate and filter config to only include fields in the dataclass
    valid_fields = {k: v for k, v in config.items() if k in type_hints}
    
    try:
        return dataclass_type(**valid_fields)
    except Exception as e:
        raise ConfigError(f"Error creating dataclass from config: {str(e)}")

def dataclass_to_config(instance: Any) -> Dict[str, Any]:
    """
    Convert a dataclass instance to a configuration dictionary.
    
    Args:
        instance: Dataclass instance
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigError: If conversion fails
    """
    if not is_dataclass(instance):
        raise ConfigError(f"Object {instance} is not a dataclass instance")
        
    try:
        return asdict(instance)
    except Exception as e:
        raise ConfigError(f"Error converting dataclass to config: {str(e)}")

def save_config(config: Dict[str, Any], file_path: Union[str, Path], format: str = 'json') -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary
        file_path: Path to save configuration
        format: File format ('json' or 'yaml')
        
    Raises:
        ConfigError: If saving fails
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
        
    try:
        with open(file_path, 'w') as f:
            if format.lower() == 'yaml':
                try:
                    import yaml
                    yaml.dump(config, f, default_flow_style=False)
                except ImportError:
                    raise ConfigError("PyYAML package not installed. Install with: pip install pyyaml")
            elif format.lower() == 'json':
                json.dump(config, f, indent=2)
            else:
                raise ConfigError(f"Unsupported config format: {format}")
    except Exception as e:
        raise ConfigError(f"Error saving config to {file_path}: {str(e)}")
        
@dataclass
class ComponentConfig:
    """Base class for component configuration."""
    
    enabled: bool = True
    debug: bool = False
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ComponentConfig':
        """Create a component config from a dictionary."""
        return config_as_dataclass(config, cls)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return dataclass_to_config(self)