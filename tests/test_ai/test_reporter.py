"""Tests for the ReporterAgent."""

import unittest
from unittest import mock

from cdas.ai.agents.reporter import ReporterAgent


class TestReporterAgent(unittest.TestCase):
    """Tests for the ReporterAgent class."""
    
    def setUp(self):
        """Set up test case."""
        self.db_session = mock.Mock()
        self.llm_manager = mock.Mock()
        
        # Mock LLM generate method
        self.llm_manager.generate.return_value = "Generated text"
        
        # Mock LLM generate_with_tools method
        self.llm_manager.generate_with_tools.return_value = {
            'content': 'Tool response',
            'tool_calls': None
        }
        
        self.reporter = ReporterAgent(self.db_session, self.llm_manager)
    
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.reporter.db_session, self.db_session)
        self.assertEqual(self.reporter.llm, self.llm_manager)
        self.assertIsNotNone(self.reporter.tools)
        self.assertIsNotNone(self.reporter.system_prompt)
    
    def test_generate_executive_summary(self):
        """Test generating an executive summary."""
        analysis_results = {
            'key_findings': [
                {'description': 'Finding 1', 'amount': 1000.0, 'confidence': 0.9},
                {'description': 'Finding 2', 'amount': 2000.0, 'confidence': 0.8}
            ],
            'suspicious_patterns': [
                {'pattern_type': 'duplicate', 'description': 'Pattern 1'},
                {'pattern_type': 'inconsistent', 'description': 'Pattern 2'}
            ],
            'disputed_amounts': [
                {'amount': 1000.0, 'description': 'Disputed amount 1'},
                {'amount': 2000.0, 'description': 'Disputed amount 2'}
            ]
        }
        
        result = self.reporter.generate_executive_summary(analysis_results)
        
        self.assertEqual(result, "Generated text")
        self.llm_manager.generate.assert_called_once()
        
        # Check that prompt contains key findings, patterns, and amounts
        call_args = self.llm_manager.generate.call_args[1]
        self.assertIn('Finding 1', call_args['prompt'])
        self.assertIn('Finding 2', call_args['prompt'])
        self.assertIn('Pattern 1', call_args['prompt'])
        self.assertIn('Pattern 2', call_args['prompt'])
        self.assertIn('Disputed amount 1', call_args['prompt'])
        self.assertIn('Disputed amount 2', call_args['prompt'])
    
    def test_generate_evidence_chain(self):
        """Test generating an evidence chain."""
        amount = 1000.0
        description = "Test description"
        document_trail = [
            {
                'doc_type': 'contract',
                'date': '2023-01-01',
                'title': 'Contract 1',
                'amount': 1000.0,
                'description': 'Contract description'
            },
            {
                'doc_type': 'invoice',
                'date': '2023-02-01',
                'title': 'Invoice 1',
                'amount': 1000.0,
                'description': 'Invoice description'
            }
        ]
        
        result = self.reporter.generate_evidence_chain(amount, description, document_trail)
        
        self.assertEqual(result, "Generated text")
        self.llm_manager.generate.assert_called_once()
        
        # Check that prompt contains amount, description, and document trail
        call_args = self.llm_manager.generate.call_args[1]
        self.assertIn('$1000.0', call_args['prompt'])
        self.assertIn('Test description', call_args['prompt'])
        self.assertIn('Contract 1', call_args['prompt'])
        self.assertIn('Invoice 1', call_args['prompt'])
    
    def test_generate_detailed_report(self):
        """Test generating a detailed report."""
        analysis_results = {
            'key_findings': [
                {'description': 'Finding 1', 'amount': 1000.0, 'confidence': 0.9},
                {'description': 'Finding 2', 'amount': 2000.0, 'confidence': 0.8}
            ]
        }
        
        result = self.reporter.generate_detailed_report(analysis_results, audience="attorney")
        
        self.assertEqual(result, "Generated text")
        self.llm_manager.generate.assert_called_once()
        
        # Check that prompt contains results and audience
        call_args = self.llm_manager.generate.call_args[1]
        self.assertIn('Finding 1', call_args['prompt'])
        self.assertIn('Finding 2', call_args['prompt'])
        self.assertIn('attorney', call_args['prompt'])
    
    def test_generate_presentation_report(self):
        """Test generating a presentation report."""
        analysis_results = {
            'key_findings': [
                {'description': 'Finding 1', 'amount': 1000.0, 'confidence': 0.9},
                {'description': 'Finding 2', 'amount': 2000.0, 'confidence': 0.8}
            ]
        }
        
        result = self.reporter.generate_presentation_report(analysis_results, audience="client")
        
        self.assertEqual(result, "Generated text")
        self.llm_manager.generate.assert_called_once()
        
        # Check that prompt contains results and audience
        call_args = self.llm_manager.generate.call_args[1]
        self.assertIn('Finding 1', call_args['prompt'])
        self.assertIn('Finding 2', call_args['prompt'])
        self.assertIn('client', call_args['prompt'])
    
    def test_generate_dispute_narrative(self):
        """Test generating a dispute narrative."""
        project_info = {
            'name': 'Test Project',
            'type': 'Construction',
            'description': 'Test description',
            'events': [
                {'date': '2023-01-01', 'description': 'Event 1'},
                {'date': '2023-02-01', 'description': 'Event 2'}
            ]
        }
        
        analysis_results = {
            'key_findings': [
                {'description': 'Finding 1', 'amount': 1000.0, 'confidence': 0.9},
                {'description': 'Finding 2', 'amount': 2000.0, 'confidence': 0.8}
            ]
        }
        
        result = self.reporter.generate_dispute_narrative(project_info, analysis_results)
        
        self.assertEqual(result, "Generated text")
        self.llm_manager.generate.assert_called_once()
        
        # Check that prompt contains project info and analysis results
        call_args = self.llm_manager.generate.call_args[1]
        self.assertIn('Test Project', call_args['prompt'])
        self.assertIn('Event 1', call_args['prompt'])
        self.assertIn('Event 2', call_args['prompt'])
        self.assertIn('Finding 1', call_args['prompt'])
        self.assertIn('Finding 2', call_args['prompt'])
    
    def test_generate_report_with_tools(self):
        """Test generating a report with tools."""
        # Mock LLM with tool calls
        self.llm_manager.generate_with_tools.return_value = {
            'content': 'Initial analysis',
            'tool_calls': [
                {
                    'function': {
                        'name': 'search_documents',
                        'arguments': '{"query": "test"}'
                    }
                }
            ]
        }
        
        # Mock additional responses
        responses = [
            {
                'content': 'More analysis',
                'tool_calls': None
            }
        ]
        self.llm_manager.generate_with_tools.side_effect = [
            {
                'content': 'Initial analysis',
                'tool_calls': [
                    {
                        'function': {
                            'name': 'search_documents',
                            'arguments': '{"query": "test"}'
                        }
                    }
                ]
            },
            {
                'content': 'More analysis',
                'tool_calls': None
            }
        ]
        
        # Mock execute_tool_calls
        self.reporter._execute_tool_calls = mock.Mock(return_value={
            'search_documents': 'Found 3 documents'
        })
        
        report_data = {
            'key_findings': [
                {'description': 'Finding 1', 'amount': 1000.0, 'confidence': 0.9}
            ]
        }
        
        result = self.reporter.generate_report_with_tools('executive_summary', report_data)
        
        self.assertEqual(result['report_type'], 'executive_summary')
        self.assertEqual(result['report_content'], 'Initial analysis\n\nMore analysis')
        self.assertEqual(result['metadata']['tool_calls'], 1)
        
        # Check that tools were used
        self.llm_manager.generate_with_tools.assert_called()
        self.reporter._execute_tool_calls.assert_called_once()
    
    def test_mock_mode(self):
        """Test report generation in mock mode."""
        # Set LLM to mock mode
        self.llm_manager.mock_mode = True
        
        report_data = {
            'key_findings': [
                {'description': 'Finding 1', 'amount': 1000.0, 'confidence': 0.9}
            ]
        }
        
        result = self.reporter.generate_report_with_tools('executive_summary', report_data)
        
        self.assertEqual(result['report_type'], 'executive_summary')
        self.assertIn('MOCK', result['report_content'])
        
        # Check that tools were not used (no tool calls made)
        self.llm_manager.generate_with_tools.assert_not_called()


if __name__ == '__main__':
    unittest.main()