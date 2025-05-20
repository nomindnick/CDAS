"""Tests for the investigator agent."""

import unittest
from unittest import mock
from typing import Dict, Any

from cdas.ai.agents.investigator import InvestigatorAgent


class TestInvestigatorAgent(unittest.TestCase):
    """Tests for the InvestigatorAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.db_session = mock.Mock()
        self.llm_manager = mock.Mock()
        self.agent = InvestigatorAgent(self.db_session, self.llm_manager)
    
    def test_initialize_tools(self):
        """Test tool initialization."""
        tools = self.agent._initialize_tools()
        
        self.assertIsInstance(tools, list)
        self.assertTrue(len(tools) > 0)
        
        # Check that each tool has required structure
        for tool in tools:
            self.assertIn('type', tool)
            self.assertEqual(tool['type'], 'function')
            self.assertIn('function', tool)
            self.assertIn('name', tool['function'])
            self.assertIn('description', tool['function'])
            self.assertIn('parameters', tool['function'])
    
    def test_investigate_success(self):
        """Test successful investigation."""
        # Mock LLM manager responses
        mock_tool_response = {
            'content': 'Investigating...',
            'tool_calls': None
        }
        mock_final_response = 'Final investigation report'
        
        self.llm_manager.generate_with_tools.return_value = mock_tool_response
        self.llm_manager.generate.return_value = mock_final_response
        
        # Run investigation
        result = self.agent.investigate('Test question', 'Test context')
        
        # Verify results
        self.assertIn('investigation_steps', result)
        self.assertIn('final_report', result)
        self.assertEqual(result['final_report'], mock_final_response)
        
        # Verify LLM calls
        self.llm_manager.generate_with_tools.assert_called_once()
        self.llm_manager.generate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
