"""Tool registry for agent tools.

This module contains the ToolRegistry class, which manages tools that can be
used by various agents in the system.
"""

import inspect
import logging
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for agent tools."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, Dict[str, Any]] = {}
    
    def register_tool(self, name: str, func: Callable, description: str, parameters: Dict[str, Any]) -> None:
        """Register a tool.
        
        Args:
            name: Tool name
            func: Tool function
            description: Tool description
            parameters: Tool parameters schema
        """
        self.tools[name] = {
            'func': func,
            'description': description,
            'parameters': parameters
        }
        logger.info(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool information or None if not found
        """
        return self.tools.get(name)
    
    def get_tool_function(self, name: str) -> Optional[Callable]:
        """Get a tool function by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool function or None if not found
        """
        tool = self.get_tool(name)
        return tool['func'] if tool else None
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                'name': name,
                'description': info['description'],
                'parameters': info['parameters']
            }
            for name, info in self.tools.items()
        ]
    
    def get_openai_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get tool definitions in OpenAI format.
        
        Returns:
            List of tool definitions in OpenAI format
        """
        return [
            {
                'type': 'function',
                'function': {
                    'name': name,
                    'description': info['description'],
                    'parameters': info['parameters']
                }
            }
            for name, info in self.tools.items()
        ]
    
    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool result
            
        Raises:
            ValueError: If tool is not found
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        
        logger.info(f"Executing tool: {name} with arguments: {arguments}")
        
        try:
            # Execute tool function
            result = tool['func'](**arguments)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {name}: {str(e)}")
            raise
