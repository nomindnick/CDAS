"""
Docstring standards and examples for the Construction Document Analysis System.

This module provides examples and guidelines for consistent docstring formatting
throughout the codebase. All modules, classes, methods, and functions should
follow these standards.

The project uses Google-style docstrings:
https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
"""

from typing import Dict, List, Any, Optional, Union, Callable


def module_docstring_example() -> None:
    """
    This is the module-level docstring.
    
    It should provide a clear overview of what the module does and any key concepts
    or components it contains. The first line (summary line) should be a complete
    sentence that fits on one line and concisely describes the module's purpose.
    
    This docstring can span multiple paragraphs. Each paragraph should be separated
    by a blank line. This is where you can provide more detailed information about
    the module's functionality and usage.
    
    Typical sections in a module docstring include:
    - Overview of the module's purpose
    - Key classes and functions
    - Usage examples
    - Important notes or warnings
    """
    pass


class ClassDocstringExample:
    """
    This is a class docstring.
    
    The class docstring should describe what the class does and its primary
    responsibilities. It should explain the purpose of the class and how it
    relates to other components in the system.
    
    Attributes:
        attribute1 (type): Description of attribute1.
        attribute2 (type): Description of attribute2.
    
    Notes:
        Any important notes about the class should be included here.
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None) -> None:
        """
        Initialize the class.
        
        Args:
            param1: Description of param1. Type is specified in the function signature.
            param2: Description of param2. Optional parameters should be noted,
                with default values mentioned when relevant.
        
        Raises:
            ValueError: If param1 is empty.
            TypeError: If param2 is not an integer.
        """
        self.attribute1 = param1
        self.attribute2 = param2
    
    def method_example(self, arg1: List[str], arg2: Dict[str, Any], 
                    arg3: Optional[bool] = False) -> Dict[str, Any]:
        """
        Short summary of what the method does.
        
        Use a blank line between the summary and the details. The method docstring
        should explain what the method does, its parameters, return values, and any
        exceptions it might raise.
        
        Args:
            arg1: Description of arg1. Type is specified in the function signature.
            arg2: Description of arg2. For more complex parameters, you can describe
                the expected structure and purpose.
            arg3: Description of arg3. For boolean flags, describe what True and False
                mean in this context.
        
        Returns:
            Dictionary containing the results. When returning complex structures,
            describe the structure and key components.
        
        Raises:
            ValueError: If arg1 is empty.
            KeyError: If required keys are missing from arg2.
        
        Examples:
            >>> instance = ClassDocstringExample("example", 42)
            >>> instance.method_example(["a", "b"], {"key": "value"})
            {"result": "success", "items": ["a", "b"]}
        """
        result = {
            "arg1": arg1,
            "arg2": arg2,
            "arg3": arg3
        }
        return result
    
    @property
    def property_example(self) -> str:
        """
        Short description of the property.
        
        Properties should be documented like methods, but without the Args section
        unless they take arguments (e.g., for a property setter).
        
        Returns:
            Description of the return value.
        
        Raises:
            Any exceptions that might be raised.
        """
        return self.attribute1


def function_example(param1: Union[str, int], 
                   param2: Optional[Callable] = None, 
                   *args: Any, 
                   **kwargs: Any) -> bool:
    """
    Short description of what the function does.
    
    Longer description that explains the function in more detail.
    It can span multiple lines and paragraphs.
    
    Args:
        param1: Description of param1. For union types, describe what
            happens for each possible type.
        param2: Description of param2. For callable parameters, describe
            the expected signature and purpose.
        *args: Description of variable positional arguments.
        **kwargs: Description of variable keyword arguments. For common
            kwargs, list them with their expected types and purposes.
    
    Returns:
        Description of the return value. Be clear about what True and
        False indicate in this context.
    
    Raises:
        TypeError: If param1 is not of an expected type.
        ValueError: If param1 does not meet validity criteria.
    
    Examples:
        >>> function_example("test")
        True
        >>> function_example(42, lambda x: x > 0)
        False
    
    Notes:
        Any additional notes or warnings about using the function.
    
    See Also:
        Reference to related functions or classes.
    """
    return True


# Docstring Guidelines:

# 1. Module Docstrings:
#    - All modules should have a docstring at the top of the file
#    - Describe the module's purpose, key components, and usage
#    - Include any important warnings or notes

# 2. Class Docstrings:
#    - All classes should have a docstring
#    - Describe the class's purpose and responsibilities
#    - List public attributes with types and descriptions
#    - Mention any base classes and what they contribute

# 3. Method and Function Docstrings:
#    - All public methods and functions should have a docstring
#    - Include sections for Args, Returns, Raises when applicable
#    - For complex methods, add Examples to demonstrate usage
#    - Document parameter types in type hints, not in docstrings

# 4. Property Docstrings:
#    - All properties should have a docstring
#    - Describe what the property represents
#    - Include Returns section (and Raises if applicable)

# 5. Private Method Docstrings:
#    - Private methods (_method) should have docstrings if they're complex
#    - Focus on implementation details relevant to maintainers

# 6. General Guidelines:
#    - Use imperative mood for the first line ("Do this", not "Does this")
#    - Keep the first line (summary) under 80 characters
#    - Separate paragraphs with blank lines
#    - Use complete sentences with proper punctuation
#    - Use reStructuredText formatting for inline markup:
#      - `code` for code references
#      - **bold** for emphasis
#      - *italic* for technical terms or variables

# 7. Type Hints:
#    - Use Python's type annotation system in the function signature
#    - Don't duplicate type information in the docstring
#    - In the Args section, focus on the meaning and purpose of parameters

# Docstring Sections Order:
# 1. Summary line
# 2. Extended description
# 3. Args
# 4. Returns
# 5. Raises
# 6. Examples
# 7. Notes
# 8. See Also