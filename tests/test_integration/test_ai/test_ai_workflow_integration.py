"""
Integration tests for complete AI workflows.

Tests end-to-end workflows involving multiple AI components working together,
simulating real-world usage scenarios of the system.
"""

import pytest
import json
from unittest import mock

from cdas.ai.agents.investigator import InvestigatorAgent
from cdas.ai.agents.reporter import ReporterAgent
from cdas.ai.semantic_search import search
from cdas.ai.semantic_search.index import VectorIndex
from cdas.ai.semantic_search.vectorizer import Vectorizer
from cdas.ai.embeddings import EmbeddingManager
from cdas.ai.llm import LLMManager
from cdas.ai.tools import document_tools
from cdas.ai.tools import financial_tools
from cdas.ai.tools import database_tools

from .test_helpers import (
    create_test_document,
    create_test_documents_with_amounts,
    setup_mock_embedding_response,
    setup_mock_completion_response,
    setup_mock_anthropic_response,
    verify_report_structure
)


class TestAIWorkflows:
    """Test end-to-end AI workflows."""

    @pytest.fixture(autouse=True)
    def setup(self, test_session, mock_openai, mock_anthropic):
        """Set up test environment."""
        self.session = test_session
        
        # Create test documents
        self.setup_test_documents()
        
        # Create AI components
        self.setup_ai_components(mock_openai, mock_anthropic)
        
        # Set up mock responses
        self.setup_mock_responses(mock_openai, mock_anthropic)
    
    def setup_test_documents(self):
        """Set up test documents for the workflow tests."""
        # Document data with potential issues
        self.doc_data = [
            {
                "doc_type": "contract",
                "party": "contractor",
                "content": """
                ORIGINAL CONTRACT
                Oak Elementary School Renovation
                
                Contractor: ABC Construction Inc.
                Owner: Woodland School District
                Date: January 15, 2023
                
                Total Contract Amount: $7,500,000.00
                
                Scope includes:
                1. New classroom wing
                2. Renovation of existing facilities
                3. Electrical upgrades throughout
                4. HVAC replacement
                """,
                "amounts": [
                    {"amount": 7500000.00, "description": "Total Contract Amount"}
                ]
            },
            {
                "doc_type": "change_order",
                "party": "contractor",
                "content": """
                CHANGE ORDER #1 (REJECTED)
                Oak Elementary School Renovation
                
                Contractor: ABC Construction Inc.
                Date: March 5, 2023
                
                Description: Additional foundation work required due to unexpected soil conditions
                Amount: $25,000.00
                
                Status: REJECTED by Owner on March 15, 2023
                Reason: Outside original scope, soil testing was contractor's responsibility
                """,
                "amounts": [
                    {"amount": 25000.00, "description": "Additional foundation work"}
                ]
            },
            {
                "doc_type": "invoice",
                "party": "contractor",
                "content": """
                INVOICE #123
                Oak Elementary School Renovation
                
                Contractor: ABC Construction Inc.
                Date: April 2, 2023
                
                Item 1: Electrical work phase 1 - $15,000.00
                Item 2: Partial foundation work - $45,000.00
                
                Total: $60,000.00
                """,
                "amounts": [
                    {"amount": 15000.00, "description": "Electrical work phase 1"},
                    {"amount": 45000.00, "description": "Partial foundation work"},
                    {"amount": 60000.00, "description": "Total"}
                ]
            },
            {
                "doc_type": "payment_app",
                "party": "contractor",
                "content": """
                PAYMENT APPLICATION #2
                Oak Elementary School Renovation
                
                Contractor: ABC Construction Inc.
                Date: May 10, 2023
                
                Previously completed work:
                1. Electrical work phase 1 - $15,000.00
                2. Partial foundation work - $45,000.00
                
                This application:
                3. Additional foundation work - $25,000.00 (As per soil conditions)
                4. Electrical work phase 1 - $15,000.00 (Remaining portion)
                
                Total this application: $40,000.00
                Total completed to date: $100,000.00
                """,
                "amounts": [
                    {"amount": 15000.00, "description": "Electrical work phase 1"},
                    {"amount": 45000.00, "description": "Partial foundation work"},
                    {"amount": 25000.00, "description": "Additional foundation work"},
                    {"amount": 15000.00, "description": "Electrical work phase 1 (Remaining portion)"},
                    {"amount": 40000.00, "description": "Total this application"},
                    {"amount": 100000.00, "description": "Total completed to date"}
                ]
            },
            {
                "doc_type": "correspondence",
                "party": "owner",
                "content": """
                MEMO
                Oak Elementary School Renovation
                
                From: Woodland School District
                To: ABC Construction Inc.
                Date: May 15, 2023
                
                Subject: Concerns with Payment Application #2
                
                We have concerns with the following items in Payment Application #2:
                
                1. The "Additional foundation work" for $25,000 was previously rejected in Change Order #1.
                   This item should not be included without an approved change order.
                
                2. The "Electrical work phase 1" appears to be billed twice, once in Invoice #123 and
                   again in this payment application.
                
                Please provide clarification before we can process this payment.
                """,
                "amounts": []
            }
        ]
        
        # Create documents in test database
        self.docs = create_test_documents_with_amounts(self.session, self.doc_data)
        
        # Add relationships between documents
        from cdas.db.models import DocumentRelationship
        
        # Get document IDs
        self.contract_id = self.docs[0].doc_id
        self.change_order_id = self.docs[1].doc_id
        self.invoice_id = self.docs[2].doc_id
        self.payment_app_id = self.docs[3].doc_id
        self.correspondence_id = self.docs[4].doc_id
        
        # Set up relationships
        relationships = [
            # Change order references contract
            DocumentRelationship(
                source_doc_id=self.change_order_id,
                target_doc_id=self.contract_id,
                relationship_type="references",
                confidence=0.95,
                meta_data={}
            ),
            # Invoice references contract
            DocumentRelationship(
                source_doc_id=self.invoice_id,
                target_doc_id=self.contract_id,
                relationship_type="references",
                confidence=0.95,
                meta_data={}
            ),
            # Payment app references contract
            DocumentRelationship(
                source_doc_id=self.payment_app_id,
                target_doc_id=self.contract_id,
                relationship_type="references",
                confidence=0.95,
                meta_data={}
            ),
            # Payment app references invoice
            DocumentRelationship(
                source_doc_id=self.payment_app_id,
                target_doc_id=self.invoice_id,
                relationship_type="references",
                confidence=0.90,
                meta_data={}
            ),
            # Payment app references change order
            DocumentRelationship(
                source_doc_id=self.payment_app_id,
                target_doc_id=self.change_order_id,
                relationship_type="references",
                confidence=0.85,
                meta_data={}
            ),
            # Correspondence references payment app
            DocumentRelationship(
                source_doc_id=self.correspondence_id,
                target_doc_id=self.payment_app_id,
                relationship_type="references",
                confidence=0.95,
                meta_data={}
            )
        ]
        
        for rel in relationships:
            self.session.add(rel)
        self.session.commit()
    
    def setup_ai_components(self, mock_openai, mock_anthropic):
        """Set up AI components for testing."""
        # Create components in mock mode
        config = {"mock_mode": True}
        
        self.llm = LLMManager(self.session, config)
        self.embedding_manager = EmbeddingManager(self.session, config)
        self.vector_index = VectorIndex()
        self.vectorizer = Vectorizer(self.embedding_manager)
        # Create a wrapper for semantic search functions
        class SearchWrapper:
            def __init__(self, session, embedding_mgr, vector_idx):
                self.session = session
                self.embedding_manager = embedding_mgr
                self.vector_index = vector_idx
                
            def search(self, query, **kwargs):
                return search.semantic_search(self.session, self.embedding_manager, query, **kwargs)
                
            def batch_embed_documents(self, **kwargs):
                return search.batch_embed_documents(self.session, self.embedding_manager, **kwargs)
                
            def semantic_query(self, query, **kwargs):
                return search.semantic_query(self.session, self.embedding_manager, query, **kwargs)
        
        self.semantic_search = SearchWrapper(self.session, self.embedding_manager, self.vector_index)
        # Create wrapper for document tools
        class DocumentToolsWrapper:
            def __init__(self, session):
                self.session = session
                
            def search_documents(self, query, **kwargs):
                return document_tools.search_documents(self.session, query, **kwargs)
                
            def get_document_content(self, doc_id):
                return document_tools.get_document_content(self.session, doc_id)
                
            def compare_documents(self, doc_id1, doc_id2):
                return document_tools.compare_documents(self.session, doc_id1, doc_id2)
        
        # Create wrapper for financial tools
        class FinancialToolsWrapper:
            def __init__(self, session):
                self.session = session
                
            def search_line_items(self, **kwargs):
                return financial_tools.search_line_items(self.session, **kwargs)
                
            def find_similar_amounts(self, amount, tolerance_percentage=0.1):
                return financial_tools.find_similar_amounts(self.session, amount, tolerance_percentage)
                
            def find_suspicious_patterns(self, pattern_type=None, min_confidence=0.5):
                return financial_tools.find_suspicious_patterns(self.session, pattern_type, min_confidence)
                
            def analyze_amount(self, amount):
                return financial_tools.analyze_amount(self.session, amount)
        
        # Create wrapper for database tools
        class DatabaseToolsWrapper:
            def __init__(self, session):
                self.session = session
                
            def run_sql_query(self, query, params=None):
                return database_tools.run_sql_query(self.session, query, params)
                
            def get_document_relationships(self, doc_id):
                return database_tools.get_document_relationships(self.session, doc_id)
                
            def get_document_metadata(self, doc_id):
                return database_tools.get_document_metadata(self.session, doc_id)
                
            def get_amount_references(self, amount, tolerance=0.01):
                return database_tools.get_amount_references(self.session, amount, tolerance)
                
            def find_date_range_activity(self, start_date, end_date, doc_type=None, party=None):
                return database_tools.find_date_range_activity(self.session, start_date, end_date, doc_type, party)
                
            def get_document_changes(self, doc_id):
                return database_tools.get_document_changes(self.session, doc_id)
                
            def get_financial_transactions(self, **kwargs):
                return database_tools.get_financial_transactions(self.session, **kwargs)
        
        self.document_tools = DocumentToolsWrapper(self.session)
        self.financial_tools = FinancialToolsWrapper(self.session)
        self.database_tools = DatabaseToolsWrapper(self.session)
        self.investigator = InvestigatorAgent(self.session, self.llm)
        self.reporter = ReporterAgent(self.session, self.llm)
        
        # Set up vector index with document content
        # Mock embedding responses for each document
        doc_embeddings = []
        doc_contents = []
        
        for i, doc in enumerate(self.docs):
            doc_content = self.doc_data[i]["content"]
            doc_contents.append(doc_content)
            doc_embeddings.append([0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)] * 512)
        
        # Add query embeddings
        doc_contents.extend([
            "electrical work duplicate billing",
            "rejected change order",
            "foundation work",
            "payment application concerns"
        ])
        doc_embeddings.extend([
            [0.15, 0.25, 0.35] * 512,
            [0.25, 0.35, 0.45] * 512,
            [0.35, 0.45, 0.55] * 512,
            [0.45, 0.55, 0.65] * 512
        ])
        
        setup_mock_embedding_response(mock_openai, doc_contents, doc_embeddings)
        
        # Add documents to vector index
        for i, doc in enumerate(self.docs):
            # Mock document retrieval for vectorization
            with mock.patch('cdas.ai.semantic_search.vectorizer.get_document_text', 
                           return_value=self.doc_data[i]["content"]):
                chunks = self.vectorizer.vectorize_document({
                    "doc_id": doc.doc_id,
                    "doc_type": doc.doc_type,
                    "party": self.doc_data[i]["party"],
                    "content": self.doc_data[i]["content"]
                })
                
                # Add chunks to index
                for chunk in chunks:
                    self.vector_index.add(chunk["vector"], chunk["metadata"])
    
    def setup_mock_responses(self, mock_openai, mock_anthropic):
        """Set up mock responses for LLMs."""
        # OpenAI mock responses
        openai_responses = {
            "duplicate billing": """
            Based on my analysis of the documents, I've identified two significant financial issues:
            
            1. DUPLICATE BILLING: The contractor has billed twice for "Electrical work phase 1" at $15,000:
               - First in Invoice #123 (April 2, 2023)
               - Again in Payment Application #2 (May 10, 2023) as "Electrical work phase 1 (Remaining portion)"
               
               There's no indication this was split into portions in the original documentation.
            
            2. REJECTED CHANGE ORDER INCLUDED: Payment Application #2 includes $25,000 for "Additional foundation work"
               which was explicitly rejected in Change Order #1 (March 15, 2023).
            
            Total disputed amount: $40,000
            
            This pattern suggests questionable billing practices that warrant further investigation.
            """,
            
            "evidence chain": """
            EVIDENCE CHAIN: $25,000 Additional Foundation Work
            
            The amount of $25,000 for "Additional foundation work" appears in the following documents:
            
            1. CHANGE ORDER #1 (doc2) - Dated: March 5, 2023
               "Description: Additional foundation work required due to unexpected soil conditions
                Amount: $25,000.00"
               Status: REJECTED by Owner on March 15, 2023
               Reason: "Outside original scope, soil testing was contractor's responsibility"
            
            2. PAYMENT APPLICATION #2 (doc4) - Dated: May 10, 2023
               "This application:
                3. Additional foundation work - $25,000.00 (As per soil conditions)"
            
            3. MEMO (doc5) - Dated: May 15, 2023
               "We have concerns with the following items in Payment Application #2:
                1. The 'Additional foundation work' for $25,000 was previously rejected in Change Order #1.
                   This item should not be included without an approved change order."
            
            ANALYSIS:
            The timeline clearly shows that the contractor submitted a change order for additional 
            foundation work which was explicitly rejected by the owner. Despite this rejection, 
            the same scope and amount appeared in Payment Application #2 approximately two months later.
            The owner immediately identified this issue in their correspondence, confirming that this
            amount remains disputed and was not approved through proper channels.
            """,
            
            "suspicious patterns": """
            SUSPICIOUS FINANCIAL PATTERNS REPORT
            
            Based on analysis of the provided documents for the Oak Elementary School Renovation project,
            the following suspicious patterns have been identified:
            
            1. REJECTED ITEMS REBILLED
               Description: Change Order #1 for foundation work ($25,000) was explicitly rejected on
               March 15, 2023, but the same amount and description reappeared in Payment Application #2
               submitted on May 10, 2023.
               Severity: HIGH
               Financial Impact: $25,000.00
            
            2. DUPLICATE BILLING
               Description: "Electrical work phase 1" for $15,000 was billed twice:
               - Initially in Invoice #123 (April 2, 2023)
               - Again in Payment Application #2 (May 10, 2023) as "remaining portion" without prior
                 documentation establishing that this work was to be billed in portions.
               Severity: HIGH
               Financial Impact: $15,000.00
            
            3. INCONSISTENT APPROVALS
               Description: Payment Application #2 includes items that reference rejected change orders,
               suggesting a pattern of attempting to obtain payment for work that was not properly approved.
               Severity: MEDIUM
               Financial Impact: $25,000.00
            
            TOTAL DISPUTED AMOUNT: $40,000.00
            
            RECOMMENDED ACTIONS:
            1. Request formal clarification from contractor on duplicate electrical billing
            2. Issue formal rejection of the $25,000 foundation work line item
            3. Review all future payment applications with heightened scrutiny
            4. Document these issues for potential dispute resolution proceedings
            """
        }
        
        # Set up mock responses
        setup_mock_completion_response(mock_openai, openai_responses)
        setup_mock_anthropic_response(mock_anthropic, openai_responses)
    
    def test_investigate_and_report_workflow(self):
        """Test complete workflow of investigation and reporting."""
        # 1. First, perform an investigation using the InvestigatorAgent
        # Mock the investigate method to simulate a full investigation
        with mock.patch('cdas.ai.agents.investigator.InvestigatorAgent.investigate') as mock_investigate:
            # Set up mock investigation results
            investigation_results = {
                "key_findings": [
                    {
                        "description": "Duplicate billing for electrical work",
                        "amount": 15000.00,
                        "confidence": 0.92,
                        "document_ids": [self.invoice_id, self.payment_app_id],
                        "evidence": "Same work billed in both Invoice #123 and Payment Application #2"
                    },
                    {
                        "description": "Rejected change order included in payment application",
                        "amount": 25000.00,
                        "confidence": 0.94,
                        "document_ids": [self.change_order_id, self.payment_app_id],
                        "evidence": "Change Order #1 was rejected but included in Payment Application #2"
                    }
                ],
                "suspicious_patterns": [
                    {
                        "pattern_type": "duplicate_billing",
                        "description": "Electrical work phase 1 billed twice",
                        "severity": "high",
                        "document_ids": [self.invoice_id, self.payment_app_id],
                        "financial_impact": 15000.00
                    },
                    {
                        "pattern_type": "rejected_items_rebilled",
                        "description": "Rejected change order items appearing in payment applications",
                        "severity": "high",
                        "document_ids": [self.change_order_id, self.payment_app_id],
                        "financial_impact": 25000.00
                    }
                ],
                "disputed_amounts": [
                    {
                        "amount": 15000.00,
                        "description": "Electrical work phase 1",
                        "status": "disputed",
                        "reason": "Duplicate billing",
                        "document_ids": [self.invoice_id, self.payment_app_id]
                    },
                    {
                        "amount": 25000.00,
                        "description": "Additional foundation work",
                        "status": "disputed",
                        "reason": "Change order not approved",
                        "document_ids": [self.change_order_id, self.payment_app_id]
                    }
                ],
                "total_disputed": 40000.00
            }
            
            # Set up the mock to return the investigation results
            mock_investigate.return_value = investigation_results
            
            # Perform the investigation (mocked)
            results = self.investigator.investigate(
                "What financial issues exist in the Oak Elementary School project?"
            )
            
            # Verify investigation results
            assert results["total_disputed"] == 40000.00
            assert len(results["key_findings"]) == 2
            
            # 2. Then, generate reports based on the investigation
            # Execute different report types
            
            # 2.1 Generate executive summary
            summary = self.reporter.generate_executive_summary(results)
            assert "duplicate billing" in summary.lower()
            assert "rejected change order" in summary.lower()
            assert "$40,000" in summary or "40,000" in summary
            
            # 2.2 Generate evidence chain for the foundation work issue
            evidence_chain = self.reporter.generate_evidence_chain(
                25000.00,
                "Additional foundation work",
                [
                    {
                        "doc_id": self.change_order_id,
                        "doc_type": "change_order",
                        "title": "Change Order #1",
                        "date": "2023-03-05",
                        "party": "contractor",
                        "status": "rejected",
                        "relevant_text": "Additional foundation work required due to unexpected soil conditions. Amount: $25,000.00",
                        "page_number": 1
                    },
                    {
                        "doc_id": self.payment_app_id,
                        "doc_type": "payment_app",
                        "title": "Payment Application #2",
                        "date": "2023-05-10",
                        "party": "contractor",
                        "status": "submitted",
                        "relevant_text": "Additional foundation work - $25,000.00 (As per soil conditions)",
                        "page_number": 1
                    }
                ]
            )
            
            assert "EVIDENCE CHAIN" in evidence_chain
            assert "Change Order #1" in evidence_chain
            assert "Payment Application #2" in evidence_chain
            assert "rejected" in evidence_chain.lower()
            
            # 2.3 Generate a detailed report
            detailed_report = self.reporter.generate_detailed_report(results, "attorney")
            
            required_elements = [
                "REPORT", "KEY FINDINGS", "SUSPICIOUS PATTERNS", "DISPUTED AMOUNTS",
                "electrical work", "foundation work", "$15,000", "$25,000"
            ]
            assert verify_report_structure(detailed_report, required_elements)
            
            # 2.4 Generate a report with tools
            # Mock the _execute_tool_calls method to return tool results
            def mock_execute_tool_calls(self, tool_calls):
                """Mock tool execution for testing."""
                results = {}
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    if function_name == "get_document_relationships":
                        results[function_name] = json.dumps({
                            "document": {"doc_id": self.payment_app_id, "doc_type": "payment_app"},
                            "relationships": [
                                {
                                    "relationship_type": "references",
                                    "direction": "outgoing",
                                    "other_doc_id": self.contract_id,
                                    "other_doc_type": "contract",
                                    "confidence": 0.95
                                },
                                {
                                    "relationship_type": "references", 
                                    "direction": "outgoing",
                                    "other_doc_id": self.invoice_id,
                                    "other_doc_type": "invoice",
                                    "confidence": 0.90
                                }
                            ]
                        })
                    elif function_name == "get_amount_references":
                        results[function_name] = json.dumps({
                            "references": [
                                {
                                    "doc_id": self.invoice_id,
                                    "doc_type": "invoice",
                                    "description": "Electrical work phase 1",
                                    "amount": 15000.00
                                },
                                {
                                    "doc_id": self.payment_app_id,
                                    "doc_type": "payment_app",
                                    "description": "Electrical work phase 1",
                                    "amount": 15000.00
                                }
                            ]
                        })
                    else:
                        results[function_name] = json.dumps({"result": "Mock result"})
                return results
            
            # Patch the tool execution method
            with mock.patch.object(ReporterAgent, '_execute_tool_calls', mock_execute_tool_calls):
                # Patch the generate_with_tools method to return expected responses
                with mock.patch.object(self.llm, 'generate_with_tools', 
                                      side_effect=[
                                          {
                                              'content': 'Initial analysis',
                                              'tool_calls': [
                                                  {
                                                      'function': {
                                                          'name': 'get_document_relationships',
                                                          'arguments': f'{{"doc_id": "{self.payment_app_id}"}}'
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
                                          {
                                              'content': """
                                              SUSPICIOUS FINANCIAL PATTERNS REPORT
                                              
                                              The analysis reveals concerning financial patterns in the Oak Elementary School Renovation project:
                                              
                                              1. DUPLICATE BILLING: $15,000
                                                 Electrical work phase 1 has been billed twice - once in Invoice #123 and again in 
                                                 Payment Application #2. This represents potential double payment for the same work.
                                              
                                              2. REJECTED CHANGE ORDER: $25,000
                                                 Change Order #1 for additional foundation work was explicitly rejected, but the same 
                                                 amount appeared in Payment Application #2 without proper approval.
                                              
                                              Document relationships confirm that Payment Application #2 references both the original 
                                              contract and Invoice #123, creating a paper trail that reveals these problematic patterns.
                                              
                                              TOTAL DISPUTED AMOUNT: $40,000
                                              """,
                                              'tool_calls': None
                                          }
                                      ]):
                    
                    # Generate report with tools
                    report_result = self.reporter.generate_report_with_tools(
                        "suspicious_patterns",
                        results
                    )
                    
                    # Verify report
                    assert report_result["report_type"] == "suspicious_patterns"
                    assert "SUSPICIOUS FINANCIAL PATTERNS REPORT" in report_result["report_content"]
                    assert "DUPLICATE BILLING" in report_result["report_content"]
                    assert "REJECTED CHANGE ORDER" in report_result["report_content"]
                    assert report_result["metadata"]["tool_calls"] == 2
    
    def test_search_and_investigate_workflow(self):
        """Test workflow of searching for documents and investigating findings."""
        # 1. First, perform semantic search to find relevant documents
        # Set up mock for semantic search
        with mock.patch('cdas.ai.semantic_search.search.semantic_search') as mock_search:
            # Mock search to return results
            mock_search.return_value = [
                {
                    "score": 0.92,
                    "metadata": {
                        "doc_id": self.payment_app_id,
                        "doc_type": "payment_app",
                        "text": "Additional foundation work - $25,000.00 (As per soil conditions)"
                    }
                },
                {
                    "score": 0.88,
                    "metadata": {
                        "doc_id": self.change_order_id,
                        "doc_type": "change_order",
                        "text": "Description: Additional foundation work required due to unexpected soil conditions. Amount: $25,000.00"
                    }
                },
                {
                    "score": 0.85,
                    "metadata": {
                        "doc_id": self.correspondence_id,
                        "doc_type": "correspondence",
                        "text": "The 'Additional foundation work' for $25,000 was previously rejected in Change Order #1."
                    }
                }
            ]
            
            # Perform search
            results = self.semantic_search.search("foundation work rejected")
            
            # Verify search results
            assert len(results) == 3
            doc_types = [r["metadata"]["doc_type"] for r in results]
            assert "payment_app" in doc_types
            assert "change_order" in doc_types
            assert "correspondence" in doc_types
            
            # 2. Use search results to feed into investigation
            # Mock the document tools to return document content
            with mock.patch('cdas.ai.tools.document_tools.get_document_content') as mock_get_content:
                # Set up to return document content when requested
                def mock_get_content_implementation(doc_id):
                    """Return content for the specified document."""
                    for i, doc in enumerate(self.docs):
                        if doc.doc_id == doc_id:
                            return json.dumps({
                                "content": self.doc_data[i]["content"],
                                "doc_type": doc.doc_type,
                                "doc_id": doc.doc_id
                            })
                    return json.dumps({"content": "Document not found", "doc_id": doc_id})
                
                mock_get_content.side_effect = mock_get_content_implementation
                
                # Mock the investigator to use search results
                with mock.patch.object(self.investigator.llm, 'generate_with_tools', 
                                      side_effect=[
                                          {
                                              'content': 'Initial analysis',
                                              'tool_calls': [
                                                  {
                                                      'function': {
                                                          'name': 'get_document_content',
                                                          'arguments': f'{{"doc_id": "{self.payment_app_id}"}}'
                                                      }
                                                  },
                                                  {
                                                      'function': {
                                                          'name': 'get_document_content',
                                                          'arguments': f'{{"doc_id": "{self.change_order_id}"}}'
                                                      }
                                                  }
                                              ]
                                          },
                                          {
                                              'content': """
                                              INVESTIGATION FINDINGS:
                                              
                                              I've identified a significant financial discrepancy involving the "Additional foundation work" 
                                              line item for $25,000.00.
                                              
                                              EVIDENCE:
                                              1. Change Order #1 (dated March 5, 2023) requested $25,000 for "Additional foundation work required
                                                 due to unexpected soil conditions"
                                              2. This change order was REJECTED by the Owner on March 15, 2023, with the reason: "Outside original
                                                 scope, soil testing was contractor's responsibility"
                                              3. Despite this rejection, the same $25,000 charge for "Additional foundation work" appeared in
                                                 Payment Application #2 (dated May 10, 2023)
                                              
                                              CONCLUSION:
                                              This represents an attempt to bill for work that was explicitly rejected, which constitutes a
                                              serious financial irregularity. The contractor has attempted to recover costs for work that was
                                              determined to be their responsibility under the original contract.
                                              
                                              RECOMMENDED ACTION:
                                              Reject the $25,000 line item in Payment Application #2 and document this issue for potential
                                              dispute resolution proceedings.
                                              """,
                                              'tool_calls': None
                                          }
                                      ]):
                    
                    # Mock the investigate method to use our mocked tools
                    with mock.patch.object(self.investigator, '_investigate_with_tools', 
                                          return_value={
                                              "findings": "INVESTIGATION FINDINGS:\n\nI've identified a significant financial discrepancy...",
                                              "key_points": ["Rejected change order reappeared in payment application"],
                                              "amounts": [25000.00],
                                              "confidence": 0.95,
                                              "document_ids": [self.change_order_id, self.payment_app_id]
                                          }):
                        
                        # Perform investigation using mocked methods
                        investigation = self.investigator.investigate(
                            "Was the foundation work charge properly approved?"
                        )
                        
                        # Verify investigation results
                        assert "findings" in investigation
                        assert "key_points" in investigation
                        assert "Rejected change order" in investigation["key_points"][0]
                        assert "amounts" in investigation
                        assert investigation["amounts"][0] == 25000.00
                        assert investigation["confidence"] > 0.9
    
    def test_mock_mode_end_to_end(self):
        """Test the entire workflow in mock mode without API calls."""
        # Create configuration with mock_mode enabled
        config = {"mock_mode": True}
        
        # Create components in mock mode
        llm = LLMManager(self.session, config)
        embedding_manager = EmbeddingManager(self.session, config)
        investigator = InvestigatorAgent(self.session, llm)
        reporter = ReporterAgent(self.session, llm)
        
        # Test the investigator in mock mode
        investigation = investigator.investigate("What suspicious patterns exist in the documents?")
        
        # Verify mock investigation results
        assert "[MOCK]" in investigation["findings"]
        assert "key_points" in investigation
        assert "amounts" in investigation
        assert "confidence" in investigation
        
        # Test report generation in mock mode
        report = reporter.generate_executive_summary({
            "key_findings": [{"description": "Test finding", "amount": 1000.0}],
            "suspicious_patterns": [],
            "disputed_amounts": []
        })
        
        # Verify mock report
        assert "[MOCK]" in report
        assert "summary" in report.lower()
        
        # Test embedding generation in mock mode
        embedding = embedding_manager.generate_embeddings("Test text")
        
        # Verify mock embedding
        assert len(embedding) == 1536
        assert embedding[0] == 0.1


def test_api_key_validation(test_session):
    """Test that API key validation works correctly."""
    # Test with valid API keys
    valid_config = {
        "openai_api_key": "sk-valid-key",
        "anthropic_api_key": "sk-anthropic-valid-key"
    }
    
    # Mock the API clients to prevent actual API calls
    with mock.patch('cdas.ai.llm.OpenAI') as mock_openai, \
         mock.patch('cdas.ai.llm.Anthropic') as mock_anthropic:
        
        # Create LLM manager with "valid" keys
        llm = LLMManager(test_session, valid_config)
        
        # Verify LLM manager initialized properly
        assert llm.openai_api_key == "sk-valid-key"
        assert llm.anthropic_api_key == "sk-anthropic-valid-key"
        assert not llm.mock_mode
        
        # Test with invalid API keys
        invalid_config = {
            "openai_api_key": "",
            "anthropic_api_key": None
        }
        
        # Create LLM manager with invalid keys (should fall back to mock mode)
        llm_invalid = LLMManager(test_session, invalid_config)
        
        # Verify LLM manager is in mock mode
        assert llm_invalid.mock_mode
        
        # Test warning when valid key is provided but mock mode is explicitly enabled
        mock_config = {
            "openai_api_key": "sk-valid-key",
            "mock_mode": True
        }
        
        # Create LLM manager with valid key but mock mode enabled
        with mock.patch('cdas.ai.llm.logging.warning') as mock_warning:
            llm_mock = LLMManager(test_session, mock_config)
            
            # Verify warning was logged
            mock_warning.assert_called_once()
            assert llm_mock.mock_mode


if __name__ == "__main__":
    pytest.main(["-v", __file__])