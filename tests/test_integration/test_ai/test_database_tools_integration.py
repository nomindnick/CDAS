"""
Integration tests for the DatabaseTools module.

Tests the DatabaseTools module's ability to execute safe queries against
the database and return structured results for AI components.
"""

import pytest
import json
from decimal import Decimal

from cdas.ai.tools.database_tools import DatabaseTools
from cdas.db.models import Document, LineItem, Page
from .test_helpers import create_test_documents_with_amounts


def test_get_amount_references(test_session):
    """Test retrieving references to a specific amount from the database."""
    # Create test documents with specific amounts
    doc_data = [
        {
            "doc_type": "invoice",
            "party": "contractor",
            "content": "Invoice for electrical work phase 1 - $15,000",
            "amounts": [
                {"amount": 15000.00, "description": "Electrical work phase 1"}
            ]
        },
        {
            "doc_type": "payment_app",
            "party": "contractor",
            "content": "Payment application including electrical work phase 1 - $15,000",
            "amounts": [
                {"amount": 15000.00, "description": "Electrical work phase 1"},
                {"amount": 25000.00, "description": "Plumbing work"}
            ]
        },
        {
            "doc_type": "change_order",
            "party": "contractor",
            "content": "Change order for additional foundation work - $25,000",
            "amounts": [
                {"amount": 25000.00, "description": "Additional foundation work"}
            ]
        }
    ]
    
    # Create documents in test database
    docs = create_test_documents_with_amounts(test_session, doc_data)
    
    # Create database tools
    db_tools = DatabaseTools(test_session)
    
    # Test exact amount match
    result_exact = db_tools.get_amount_references(15000.00)
    result_dict = json.loads(result_exact)
    
    # Verify results
    assert "references" in result_dict
    assert len(result_dict["references"]) == 2
    
    # Check that both documents with $15,000 are found
    doc_types_found = [ref["doc_type"] for ref in result_dict["references"]]
    assert "invoice" in doc_types_found
    assert "payment_app" in doc_types_found
    
    # Test with tolerance
    result_with_tolerance = db_tools.get_amount_references(15005.00, tolerance=100.00)
    result_dict_tolerance = json.loads(result_with_tolerance)
    
    # Verify results
    assert "references" in result_dict_tolerance
    assert len(result_dict_tolerance["references"]) == 2


def test_get_document_amounts(test_session):
    """Test retrieving all amounts from a specific document."""
    # Create test document with multiple amounts
    doc_data = [
        {
            "doc_type": "payment_app",
            "party": "contractor",
            "content": "Payment application with multiple items",
            "amounts": [
                {"amount": 15000.00, "description": "Electrical work phase 1"},
                {"amount": 25000.00, "description": "Plumbing work"},
                {"amount": 35000.00, "description": "HVAC installation"},
                {"amount": 75000.00, "description": "Total this application"}
            ]
        }
    ]
    
    # Create document in test database
    docs = create_test_documents_with_amounts(test_session, doc_data)
    doc_id = docs[0].doc_id
    
    # Create database tools
    db_tools = DatabaseTools(test_session)
    
    # Test get_document_amounts
    result = db_tools.get_document_amounts(doc_id)
    result_dict = json.loads(result)
    
    # Verify results
    assert "document" in result_dict
    assert "amounts" in result_dict
    assert len(result_dict["amounts"]) == 4
    
    # Check that all amounts are included
    amounts = [Decimal(str(amount["amount"])) for amount in result_dict["amounts"]]
    assert Decimal('15000.00') in amounts
    assert Decimal('25000.00') in amounts
    assert Decimal('35000.00') in amounts
    assert Decimal('75000.00') in amounts


def test_find_related_documents(test_session):
    """Test finding documents related to a specific document."""
    # Create test documents with related metadata
    doc_data = [
        {
            "doc_type": "contract",
            "party": "contractor",
            "content": "Original contract for project",
            "amounts": [
                {"amount": 500000.00, "description": "Total contract amount"}
            ]
        },
        {
            "doc_type": "change_order",
            "party": "contractor",
            "content": "Change order referencing contract",
            "amounts": [
                {"amount": 25000.00, "description": "Additional work"}
            ]
        },
        {
            "doc_type": "payment_app",
            "party": "contractor",
            "content": "Payment application for completed work",
            "amounts": [
                {"amount": 100000.00, "description": "Work completed to date"}
            ]
        }
    ]
    
    # Create documents in test database
    docs = create_test_documents_with_amounts(test_session, doc_data)
    
    # Set up relationships between documents
    contract_id = docs[0].doc_id
    change_order_id = docs[1].doc_id
    payment_app_id = docs[2].doc_id
    
    # Update metadata to include references
    docs[1].meta_data = {"references": [contract_id], "project_id": "test_project"}
    docs[2].meta_data = {"references": [contract_id], "project_id": "test_project"}
    
    # Add explicit relationship records (would typically be done by analysis engine)
    from cdas.db.models import DocumentRelationship
    
    # Change order references contract
    co_rel = DocumentRelationship(
        source_doc_id=change_order_id,
        target_doc_id=contract_id,
        relationship_type="references",
        confidence=0.95,
        meta_data={}
    )
    
    # Payment app references contract
    pa_rel = DocumentRelationship(
        source_doc_id=payment_app_id,
        target_doc_id=contract_id,
        relationship_type="references",
        confidence=0.95,
        meta_data={}
    )
    
    test_session.add(co_rel)
    test_session.add(pa_rel)
    test_session.commit()
    
    # Create database tools
    db_tools = DatabaseTools(test_session)
    
    # Test find_related_documents
    result = db_tools.find_related_documents(contract_id)
    result_dict = json.loads(result)
    
    # Verify results
    assert "document" in result_dict
    assert "related_documents" in result_dict
    assert len(result_dict["related_documents"]) == 2
    
    # Check that both related documents are found
    related_ids = [doc["doc_id"] for doc in result_dict["related_documents"]]
    assert change_order_id in related_ids
    assert payment_app_id in related_ids


def test_run_sql_query(test_session):
    """Test running safe SQL queries against the database."""
    # Create test documents with various doc types
    doc_data = [
        {"doc_type": "invoice", "party": "contractor", "content": "Invoice 1", 
         "amounts": [{"amount": 10000.00, "description": "Item 1"}]},
        {"doc_type": "invoice", "party": "contractor", "content": "Invoice 2", 
         "amounts": [{"amount": 20000.00, "description": "Item 2"}]},
        {"doc_type": "payment_app", "party": "contractor", "content": "Payment App 1", 
         "amounts": [{"amount": 30000.00, "description": "Item 3"}]},
        {"doc_type": "change_order", "party": "contractor", "content": "Change Order 1", 
         "amounts": [{"amount": 40000.00, "description": "Item 4"}]},
        {"doc_type": "change_order", "party": "owner", "content": "Change Order 2", 
         "amounts": [{"amount": 50000.00, "description": "Item 5"}]}
    ]
    
    # Create documents in test database
    docs = create_test_documents_with_amounts(test_session, doc_data)
    
    # Create database tools
    db_tools = DatabaseTools(test_session)
    
    # Test simple count query
    count_query = "SELECT doc_type, COUNT(*) as count FROM documents GROUP BY doc_type"
    result_count = db_tools.run_sql_query(count_query)
    result_count_dict = json.loads(result_count)
    
    # Verify count results
    assert "results" in result_count_dict
    counts = {r["doc_type"]: r["count"] for r in result_count_dict["results"]}
    assert counts["invoice"] == 2
    assert counts["payment_app"] == 1
    assert counts["change_order"] == 2
    
    # Test query with parameters
    param_query = "SELECT * FROM documents WHERE doc_type = :doc_type AND party = :party"
    params = {"doc_type": "change_order", "party": "owner"}
    result_params = db_tools.run_sql_query(param_query, params)
    result_params_dict = json.loads(result_params)
    
    # Verify parameterized query results
    assert "results" in result_params_dict
    assert len(result_params_dict["results"]) == 1
    assert result_params_dict["results"][0]["doc_type"] == "change_order"
    assert result_params_dict["results"][0]["party"] == "owner"
    
    # Test complex join query
    join_query = """
    SELECT d.doc_id, d.doc_type, li.amount, li.description
    FROM documents d
    JOIN line_items li ON d.doc_id = li.doc_id
    WHERE li.amount > :min_amount
    ORDER BY li.amount DESC
    """
    join_params = {"min_amount": 30000.00}
    result_join = db_tools.run_sql_query(join_query, join_params)
    result_join_dict = json.loads(result_join)
    
    # Verify join query results
    assert "results" in result_join_dict
    assert len(result_join_dict["results"]) == 2  # Should find the 40K and 50K amounts
    assert float(result_join_dict["results"][0]["amount"]) == 50000.00
    assert float(result_join_dict["results"][1]["amount"]) == 40000.00


def test_find_suspicious_amounts(test_session):
    """Test finding suspicious amounts across documents."""
    # Create test documents with suspicious patterns
    doc_data = [
        {
            "doc_type": "change_order",
            "party": "contractor",
            "content": "Change order - rejected",
            "amounts": [
                {"amount": 25000.00, "description": "Additional foundation work", 
                 "status": "rejected"}
            ]
        },
        {
            "doc_type": "payment_app",
            "party": "contractor",
            "content": "Payment application with rejected item",
            "amounts": [
                {"amount": 25000.00, "description": "Foundation work", "status": "pending"},
                {"amount": 15000.00, "description": "Electrical work", "status": "pending"}
            ]
        },
        {
            "doc_type": "invoice",
            "party": "contractor",
            "content": "Invoice for electrical work",
            "amounts": [
                {"amount": 15000.00, "description": "Electrical work phase 1", 
                 "status": "pending"}
            ]
        },
        {
            "doc_type": "payment_app",
            "party": "contractor",
            "content": "Another payment application with same electrical work",
            "amounts": [
                {"amount": 15000.00, "description": "Electrical work phase 1", 
                 "status": "pending"},
                {"amount": 35000.00, "description": "HVAC installation", "status": "pending"}
            ]
        }
    ]
    
    # Create documents in test database
    docs = create_test_documents_with_amounts(test_session, doc_data)
    
    # Create database tools
    db_tools = DatabaseTools(test_session)
    
    # Test find_suspicious_amounts
    result = db_tools.find_suspicious_amounts()
    result_dict = json.loads(result)
    
    # Verify results
    assert "suspicious_amounts" in result_dict
    
    # Should find both suspicious patterns: rejected change order and duplicate billing
    found_rejected = False
    found_duplicate = False
    
    for suspicious in result_dict["suspicious_amounts"]:
        if suspicious["amount"] == 25000.00 and suspicious["pattern"] == "rejected_item_rebilled":
            found_rejected = True
        if suspicious["amount"] == 15000.00 and suspicious["pattern"] == "duplicate_billing":
            found_duplicate = True
    
    assert found_rejected, "Should find rejected change order item reappearing in payment app"
    assert found_duplicate, "Should find duplicate billing for electrical work"


def test_get_document_relationships(test_session):
    """Test retrieving document relationships from the database."""
    # Create test documents
    doc_data = [
        {"doc_type": "contract", "party": "contractor", "content": "Original contract",
         "amounts": [{"amount": 500000.00, "description": "Total contract amount"}]},
        {"doc_type": "change_order", "party": "contractor", "content": "Change order 1",
         "amounts": [{"amount": 25000.00, "description": "Additional work"}]},
        {"doc_type": "change_order", "party": "contractor", "content": "Change order 2",
         "amounts": [{"amount": 15000.00, "description": "More work"}]},
    ]
    
    # Create documents in test database
    docs = create_test_documents_with_amounts(test_session, doc_data)
    
    # Get document IDs
    contract_id = docs[0].doc_id
    co1_id = docs[1].doc_id
    co2_id = docs[2].doc_id
    
    # Add relationships
    from cdas.db.models import DocumentRelationship
    
    relationships = [
        # Both change orders reference the contract
        DocumentRelationship(
            source_doc_id=co1_id,
            target_doc_id=contract_id,
            relationship_type="references",
            confidence=0.95,
            meta_data={"section": "general conditions"}
        ),
        DocumentRelationship(
            source_doc_id=co2_id,
            target_doc_id=contract_id,
            relationship_type="references",
            confidence=0.90,
            meta_data={"section": "scope of work"}
        ),
        # Change order 2 supersedes change order 1
        DocumentRelationship(
            source_doc_id=co2_id,
            target_doc_id=co1_id,
            relationship_type="supersedes",
            confidence=0.85,
            meta_data={"reason": "revised scope"}
        )
    ]
    
    for rel in relationships:
        test_session.add(rel)
    test_session.commit()
    
    # Create database tools
    db_tools = DatabaseTools(test_session)
    
    # Test get_document_relationships for contract
    result_contract = db_tools.get_document_relationships(contract_id)
    result_contract_dict = json.loads(result_contract)
    
    # Verify contract relationships (should have incoming references)
    assert "document" in result_contract_dict
    assert "relationships" in result_contract_dict
    assert len(result_contract_dict["relationships"]) == 2
    
    # All relationships should be incoming references to the contract
    for rel in result_contract_dict["relationships"]:
        assert rel["relationship_type"] == "references"
        assert rel["direction"] == "incoming"
        assert rel["other_doc_id"] in [co1_id, co2_id]
    
    # Test get_document_relationships for change order 2
    result_co2 = db_tools.get_document_relationships(co2_id)
    result_co2_dict = json.loads(result_co2)
    
    # Verify change order 2 relationships (should have outgoing reference and supersedes)
    assert "document" in result_co2_dict
    assert "relationships" in result_co2_dict
    assert len(result_co2_dict["relationships"]) == 2
    
    # Check for both outgoing relationships
    outgoing_types = [rel["relationship_type"] for rel in result_co2_dict["relationships"]
                     if rel["direction"] == "outgoing"]
    assert "references" in outgoing_types
    assert "supersedes" in outgoing_types


def test_database_tools_security(test_session):
    """Test security features of database tools to prevent SQL injection."""
    # Create database tools
    db_tools = DatabaseTools(test_session)
    
    # Test prevention of destructive queries
    dangerous_queries = [
        "DROP TABLE documents",
        "DELETE FROM documents",
        "UPDATE documents SET status = 'deleted'",
        "INSERT INTO documents (doc_id) VALUES ('fake')",
        "ALTER TABLE documents ADD COLUMN hack TEXT",
        "CREATE TABLE hack (id INTEGER)",
        "PRAGMA table_info(documents)"
    ]
    
    for query in dangerous_queries:
        with pytest.raises(ValueError, match="only SELECT queries are allowed"):
            db_tools.run_sql_query(query)
    
    # Test prevention of multiple statements
    multi_statement = "SELECT * FROM documents; DELETE FROM documents"
    with pytest.raises(ValueError, match="only one statement is allowed"):
        db_tools.run_sql_query(multi_statement)
    
    # Test parameterized queries for security
    # This would be dangerous if parameters weren't handled properly
    safe_query = "SELECT * FROM documents WHERE doc_type = :doc_type"
    safe_result = db_tools.run_sql_query(safe_query, {"doc_type": "'; DROP TABLE documents; --"})
    
    # Should execute safely with no records returned (since the injection is treated as a literal)
    safe_result_dict = json.loads(safe_result)
    assert "results" in safe_result_dict
    assert len(safe_result_dict["results"]) == 0  # No matching records


if __name__ == "__main__":
    pytest.main(["-v", __file__])