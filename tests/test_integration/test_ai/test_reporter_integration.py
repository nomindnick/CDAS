"""
Integration tests for the ReporterAgent.

Tests the ReporterAgent's ability to generate different types of reports
based on analysis results and project data.
"""

import pytest
from unittest import mock

from cdas.ai.agents.reporter import ReporterAgent
from cdas.ai.llm import LLMManager
from .test_helpers import (
    verify_report_structure,
    verify_evidence_chain,
    setup_mock_completion_response
)


def test_reporter_executive_summary(test_session, llm_manager, sample_analysis_results, mock_openai):
    """Test that reporter can generate an executive summary from analysis results."""
    # Set up mock responses
    mock_responses = {
        "executive summary": """
        Executive Summary: Financial Analysis Findings
        
        The analysis has identified two significant financial issues with a total disputed amount of $40,000. 
        First, electrical work valued at $15,000 was billed twice, appearing in both invoice #123 and payment 
        application #4. Second, a previously rejected change order for additional foundation work ($25,000) 
        was improperly included in payment application #5. Both issues are supported by strong documentary evidence,
        showing a pattern of questionable billing practices that warrant immediate attention.
        """
    }
    setup_mock_completion_response(mock_openai, mock_responses)
    
    # Create reporter agent
    reporter = ReporterAgent(test_session, llm_manager)
    
    # Generate executive summary
    summary = reporter.generate_executive_summary(sample_analysis_results)
    
    # Verify summary structure
    required_elements = ["financial", "analysis", "$15,000", "$25,000", "electrical work", "foundation work"]
    assert verify_report_structure(summary, required_elements)


def test_reporter_evidence_chain(test_session, llm_manager, sample_document_trail, mock_openai):
    """Test that reporter can generate an evidence chain for a specific amount."""
    # Set up mock responses
    mock_responses = {
        "evidence chain": """
        EVIDENCE CHAIN: $15,000 Electrical Work Phase 1
        
        The amount of $15,000 for "Electrical work phase 1" appears in the following documents:
        
        1. ORIGINAL CONTRACT (doc1) - Dated: 2023-01-15
           "Electrical work phase 1 to be completed for $15,000"
           Location: Page 8
        
        2. INVOICE #123 (doc2) - Dated: 2023-02-20
           "Electrical work phase 1 - $15,000"
           Location: Page 1
        
        3. PAYMENT APPLICATION #4 (doc4) - Dated: 2023-03-10
           "Electrical work phase 1 - $15,000"
           Location: Page 2
        
        ANALYSIS:
        This amount appears in three separate documents spanning from the original contract to 
        a payment application, indicating potential duplicate billing. The same scope of work 
        appears to have been billed multiple times without clear justification for the repetition.
        """
    }
    setup_mock_completion_response(mock_openai, mock_responses)
    
    # Create reporter agent
    reporter = ReporterAgent(test_session, llm_manager)
    
    # Generate evidence chain
    evidence_chain = reporter.generate_evidence_chain(
        15000.00,
        "Electrical work phase 1",
        sample_document_trail
    )
    
    # Verify evidence chain
    doc_ids = ["doc1", "doc2", "doc4"]
    assert verify_evidence_chain(evidence_chain, 15000.00, doc_ids)


def test_reporter_detailed_report(test_session, llm_manager, sample_analysis_results, mock_openai):
    """Test that reporter can generate a detailed report for an attorney."""
    # Set up mock responses
    mock_responses = {
        "detailed report": """
        DETAILED FINANCIAL ANALYSIS REPORT
        Prepared for: Attorney
        
        1. EXECUTIVE SUMMARY
           This report details financial irregularities totaling $40,000 in the Oak Elementary 
           School Renovation project, supported by documentary evidence.
        
        2. KEY FINDINGS
           2.1. Duplicate Billing for Electrical Work
                Amount: $15,000
                Confidence: High (92%)
                Evidence: Same work billed in both invoice #123 and payment application #4
           
           2.2. Unapproved Change Order Included in Payment
                Amount: $25,000
                Confidence: High (89%)
                Evidence: Change order #7 was rejected but included in payment application #5
        
        3. SUSPICIOUS PATTERNS
           3.1. Multiple instances of duplicate billing
           3.2. Rejected items consistently reappearing in payment applications
        
        4. EVIDENCE ANALYSIS
           [Detailed analysis of each document and finding...]
        
        5. RECOMMENDATIONS
           [Strategic recommendations for dispute resolution...]
        """
    }
    setup_mock_completion_response(mock_openai, mock_responses)
    
    # Create reporter agent
    reporter = ReporterAgent(test_session, llm_manager)
    
    # Generate detailed report
    report = reporter.generate_detailed_report(sample_analysis_results, "attorney")
    
    # Verify report structure
    required_elements = [
        "DETAILED", "REPORT", "EXECUTIVE SUMMARY", "KEY FINDINGS", 
        "Duplicate Billing", "$15,000", "Unapproved Change Order",
        "$25,000", "EVIDENCE", "RECOMMENDATIONS"
    ]
    assert verify_report_structure(report, required_elements)


def test_reporter_with_tools(test_session, llm_manager, sample_analysis_results, mock_openai):
    """Test that reporter can generate a report using tools to gather additional data."""
    # Set up mock for tool execution
    original_execute_tool_calls = ReporterAgent._execute_tool_calls
    
    def mock_execute_tool_calls(self, tool_calls):
        """Mock tool execution to return predefined results."""
        results = {}
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            if function_name == "search_documents":
                results[function_name] = {
                    "documents": [
                        {"doc_id": "doc1", "title": "Original Contract", "content": "Sample content"},
                        {"doc_id": "doc2", "title": "Invoice #123", "content": "Sample content"}
                    ]
                }
            elif function_name == "get_amount_references":
                results[function_name] = {
                    "references": [
                        {"doc_id": "doc1", "description": "Electrical work", "amount": 15000.00},
                        {"doc_id": "doc2", "description": "Electrical work", "amount": 15000.00}
                    ]
                }
            else:
                results[function_name] = {"result": "Mock result for " + function_name}
        return results
    
    # Patch the execute_tool_calls method
    with mock.patch.object(ReporterAgent, '_execute_tool_calls', mock_execute_tool_calls):
        # Set up mock LLM responses with tool calls
        llm_responses = [
            # First response with tool calls
            {
                'content': 'Initial analysis',
                'tool_calls': [
                    {
                        'function': {
                            'name': 'search_documents',
                            'arguments': '{"query": "electrical work"}'
                        }
                    },
                    {
                        'function': {
                            'name': 'get_amount_references',
                            'arguments': '{"amount": 15000.0, "tolerance": 0.01}'
                        }
                    }
                ]
            },
            # Second response after tool results
            {
                'content': """
                COMPREHENSIVE FINANCIAL ANALYSIS REPORT
                
                Based on the analysis of financial records and supporting documentation,
                this report identifies significant billing irregularities totaling $40,000.
                
                1. DUPLICATE BILLING: $15,000
                   The electrical work billed for $15,000 appears in multiple documents,
                   including the original contract (doc1) and invoice #123 (doc2).
                
                2. UNAPPROVED CHARGES: $25,000
                   A previously rejected change order for foundation work was improperly
                   included in subsequent payment applications.
                
                CONCLUSION:
                The evidence establishes a clear pattern of questionable billing practices
                that merit further investigation and potential dispute resolution.
                """,
                'tool_calls': None
            }
        ]
        
        # Patch the LLM's generate_with_tools method
        with mock.patch.object(llm_manager, 'generate_with_tools', 
                              side_effect=llm_responses):
            # Create reporter agent
            reporter = ReporterAgent(test_session, llm_manager)
            
            # Generate report with tools
            result = reporter.generate_report_with_tools(
                "detailed_analysis",
                sample_analysis_results
            )
            
            # Verify result structure
            assert result["report_type"] == "detailed_analysis"
            assert "COMPREHENSIVE FINANCIAL ANALYSIS REPORT" in result["report_content"]
            assert "DUPLICATE BILLING: $15,000" in result["report_content"]
            assert "UNAPPROVED CHARGES: $25,000" in result["report_content"]
            assert result["metadata"]["tool_calls"] == 2


def test_reporter_dispute_narrative(test_session, llm_manager, sample_project_info, 
                                  sample_analysis_results, mock_openai):
    """Test that reporter can generate a dispute narrative combining project info and analysis."""
    # Set up mock responses
    mock_responses = {
        "dispute narrative": """
        OAK ELEMENTARY SCHOOL RENOVATION: DISPUTE NARRATIVE
        
        PROJECT OVERVIEW:
        The Oak Elementary School Renovation project involves the renovation of an existing 
        elementary school including a new classroom wing and modernized facilities. The contract 
        between Woodland School District and ABC Construction Inc. was executed on January 15, 2023,
        with a total contract value of $7,500,000 and scheduled completion by June 30, 2024.
        
        DISPUTE CHRONOLOGY:
        1. January 15, 2023: Contract signed between Woodland School District and ABC Construction
        2. February 10, 2023: Site mobilization completed
        3. March 5, 2023: Change Order #1 submitted for additional foundation work ($25,000)
        4. March 15, 2023: Change Order #1 rejected by District
        5. April 10, 2023: Payment Application #1 submitted and approved
        6. May 12, 2023: District discovers previously rejected change order items reappearing
           in Payment Application #2
        
        FINANCIAL ANALYSIS FINDINGS:
        Our analysis identified two significant financial irregularities:
        
        1. Duplicate Billing for Electrical Work ($15,000)
           The same electrical work was billed twice, appearing in both the original invoice
           and a subsequent payment application.
           
        2. Rejected Change Order Items ($25,000)
           Change Order #1 for additional foundation work was rejected on March 15, 2023,
           but the same scope and amount reappeared in Payment Application #2.
        
        CONCLUSION:
        The timeline and financial analysis demonstrate a pattern of problematic billing practices.
        The contractor has attempted to recover costs for previously rejected work and has
        submitted duplicate billings for the same scope of work. The total financial impact
        of these issues is $40,000, which should be contested by the District.
        """
    }
    setup_mock_completion_response(mock_openai, mock_responses)
    
    # Create reporter agent
    reporter = ReporterAgent(test_session, llm_manager)
    
    # Generate dispute narrative
    narrative = reporter.generate_dispute_narrative(sample_project_info, sample_analysis_results)
    
    # Verify narrative structure
    required_elements = [
        "OAK ELEMENTARY", "RENOVATION", "DISPUTE", "PROJECT OVERVIEW",
        "DISPUTE CHRONOLOGY", "FINANCIAL ANALYSIS", "$15,000", "$25,000",
        "Change Order", "rejected", "pattern"
    ]
    assert verify_report_structure(narrative, required_elements)


def test_mock_mode_operation(test_session):
    """Test that reporter operates correctly in mock mode without API calls."""
    # Create LLM manager in mock mode
    config = {"mock_mode": True}
    llm = LLMManager(test_session, config)
    
    # Create reporter agent
    reporter = ReporterAgent(test_session, llm)
    
    # Generate executive summary in mock mode
    summary = reporter.generate_executive_summary({
        "key_findings": [{"description": "Test finding", "amount": 1000.0}],
        "suspicious_patterns": [],
        "disputed_amounts": []
    })
    
    # Verify mock response format
    assert "[MOCK]" in summary
    assert "executive summary" in summary.lower()


if __name__ == "__main__":
    pytest.main(["-v", __file__])