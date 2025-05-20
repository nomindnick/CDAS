"""
Dependency injection utilities for the Construction Document Analysis System.

This module provides a simple dependency injection container for managing
component creation and dependencies throughout the system.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable, TypeVar, Generic, List, Set, cast
import inspect

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')

class DependencyError(Exception):
    """Exception raised for dependency errors."""
    pass

class Container:
    """Simple dependency injection container."""
    
    def __init__(self):
        """Initialize the container."""
        self._factories: Dict[str, Callable[..., Any]] = {}
        self._instances: Dict[str, Any] = {}
        self._dependencies: Dict[str, Set[str]] = {}
    
    def register(self, name: str, factory: Callable[..., Any]) -> None:
        """
        Register a component factory.
        
        Args:
            name: Component name
            factory: Factory function or class constructor
            
        Raises:
            DependencyError: If component already registered
        """
        if name in self._factories:
            raise DependencyError(f"Component '{name}' already registered")
            
        self._factories[name] = factory
        
        # Inspect factory to find dependencies
        sig = inspect.signature(factory)
        self._dependencies[name] = set()
        
        for param_name, param in sig.parameters.items():
            # Skip args and kwargs
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
                
            # Skip self
            if param_name == 'self':
                continue
                
            # Add as dependency if registered or starts with underscore (internal)
            if param_name in self._factories or param_name.startswith('_'):
                self._dependencies[name].add(param_name)
    
    def register_instance(self, name: str, instance: Any) -> None:
        """
        Register a pre-created instance.
        
        Args:
            name: Component name
            instance: Component instance
            
        Raises:
            DependencyError: If component already registered
        """
        if name in self._instances or name in self._factories:
            raise DependencyError(f"Component '{name}' already registered")
            
        self._instances[name] = instance
        self._dependencies[name] = set()
    
    def get(self, name: str, **kwargs) -> Any:
        """
        Get a component instance.
        
        Args:
            name: Component name
            **kwargs: Override parameters for factory
            
        Returns:
            Component instance
            
        Raises:
            DependencyError: If component not registered or dependency cycle detected
        """
        # If instance already exists, return it
        if name in self._instances:
            return self._instances[name]
            
        # Check if factory exists
        if name not in self._factories:
            raise DependencyError(f"Component '{name}' not registered")
            
        # Check for circular dependencies
        stack = self._check_circular_dependencies(name, [])
        if stack:
            cycle = ' -> '.join(stack + [name])
            raise DependencyError(f"Circular dependency detected: {cycle}")
            
        # Resolve dependencies
        deps = {}
        for dep_name in self._dependencies[name]:
            if dep_name.startswith('_'):
                # Special dependency, skip
                continue
            elif dep_name in kwargs:
                # Use provided override
                deps[dep_name] = kwargs[dep_name]
            else:
                # Recursively resolve dependency
                deps[dep_name] = self.get(dep_name)
                
        # Create instance
        factory = self._factories[name]
        instance = factory(**deps, **{k: v for k, v in kwargs.items() if k not in deps})
        
        # Cache instance
        self._instances[name] = instance
        
        return instance
    
    def _check_circular_dependencies(self, name: str, stack: List[str]) -> List[str]:
        """
        Check for circular dependencies.
        
        Args:
            name: Component name
            stack: Current dependency stack
            
        Returns:
            Dependency cycle if detected, empty list otherwise
        """
        if name in stack:
            return stack[stack.index(name):]
            
        new_stack = stack + [name]
        
        for dep_name in self._dependencies.get(name, set()):
            if dep_name.startswith('_'):
                continue
                
            cycle = self._check_circular_dependencies(dep_name, new_stack)
            if cycle:
                return cycle
                
        return []
    
    def clear(self) -> None:
        """Clear all registered components and instances."""
        self._factories.clear()
        self._instances.clear()
        self._dependencies.clear()
    
    def clear_instances(self) -> None:
        """Clear only cached instances, keeping factories."""
        self._instances.clear()


class Registry:
    """Registry for component factories."""
    
    def __init__(self):
        """Initialize the registry."""
        self._factories: Dict[str, Dict[str, Callable[..., Any]]] = {}
    
    def register(self, component_type: str, name: str, factory: Callable[..., Any]) -> None:
        """
        Register a component factory.
        
        Args:
            component_type: Component type (e.g., 'extractor', 'analyzer')
            name: Component name
            factory: Factory function or class constructor
            
        Raises:
            DependencyError: If component already registered
        """
        if component_type not in self._factories:
            self._factories[component_type] = {}
            
        if name in self._factories[component_type]:
            raise DependencyError(f"Component '{name}' already registered for type '{component_type}'")
            
        self._factories[component_type][name] = factory
    
    def get_factory(self, component_type: str, name: str) -> Optional[Callable[..., Any]]:
        """
        Get a component factory.
        
        Args:
            component_type: Component type
            name: Component name
            
        Returns:
            Factory function or None if not found
        """
        if component_type not in self._factories:
            return None
            
        return self._factories[component_type].get(name)
    
    def get_all_factories(self, component_type: str) -> Dict[str, Callable[..., Any]]:
        """
        Get all factories for a component type.
        
        Args:
            component_type: Component type
            
        Returns:
            Dictionary of factories
        """
        return self._factories.get(component_type, {}).copy()
    
    def list_components(self, component_type: str) -> List[str]:
        """
        List all registered components of a type.
        
        Args:
            component_type: Component type
            
        Returns:
            List of component names
        """
        return list(self._factories.get(component_type, {}).keys())
    
    def clear(self) -> None:
        """Clear all registered factories."""
        self._factories.clear()


# Singleton container instance for global dependency management
container = Container()

# Singleton registry instance for global component registration
registry = Registry()