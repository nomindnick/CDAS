"""
Integration tests for financial analysis workflows.

These tests verify the end-to-end functionality of financial analysis,
including pattern detection, anomaly detection, and relationship tracking.
"""

import os
import pytest
from decimal import Decimal

from cdas.financial_analysis.engine import FinancialAnalysisEngine
from cdas.db.models import Document, LineItem, FinancialTransaction, AnalysisFlag
from cdas.db.operations import get_document_by_id, register_document


def test_financial_analysis_engine_initialization(test_session):
    """Test that the financial analysis engine initializes correctly."""
    engine = FinancialAnalysisEngine(test_session)
    
    assert engine is not None
    assert engine.db_session == test_session


def test_analyze_document_patterns(test_session, sample_payment_app, sample_change_order):
    """Test analyzing patterns within and across documents."""
    payment_app, _ = sample_payment_app
    change_order, _ = sample_change_order
    
    # Create the analysis engine
    engine = FinancialAnalysisEngine(test_session)
    
    # Analyze the payment application
    payment_results = engine.analyze_document(payment_app.doc_id)
    
    # Verify the results
    assert payment_results is not None
    assert "patterns" in payment_results
    assert "anomalies" in payment_results
    assert "relationships" in payment_results
    
    # Analyze the change order
    change_order_results = engine.analyze_document(change_order.doc_id)
    
    # Verify the results
    assert change_order_results is not None
    assert "patterns" in change_order_results
    assert "anomalies" in change_order_results
    assert "relationships" in change_order_results


def test_detect_matching_amounts_across_documents(test_session, sample_payment_app, sample_change_order, request):
    """Test detecting matching amounts across different document types."""
    # Create a new document with a line item that matches an existing amount
    # (electrical work for $35,000.00 in payment_app)
    test_name = request.node.name
    unique_path = f"/test/invoice_{test_name}_{os.getpid()}.txt"
    doc = register_document(
        test_session,
        file_path=unique_path,
        doc_type="invoice",
        party="contractor",
        metadata={"project_id": "test_project"}
    )
    
    # Add a line item with a matching amount
    invoice_item = LineItem(
        doc_id=doc.doc_id,
        item_id=f"invoice_item_1_{test_name}_{os.getpid()}",
        item_number="1",
        description="Electrical System Installation",
        amount=35000.00,  # Same as in payment_app
        category="electrical",
        status="pending",
        metadata={}
    )
    
    test_session.add(invoice_item)
    test_session.commit()
    
    # Make sure to query the database to confirm our line items exist
    existing_electrical = test_session.query(LineItem).filter(
        LineItem.amount == 35000.00,
        LineItem.category == "electrical"
    ).all()
    print(f"Found {len(existing_electrical)} electrical items with amount 35000.00")
    for item in existing_electrical:
        print(f"  - {item.item_id}: {item.description} (${item.amount}) in doc {item.doc_id}")
    
    # Run financial analysis with a lower confidence threshold
    engine = FinancialAnalysisEngine(test_session)
    results = engine.find_matching_amounts(min_confidence=0.5)
    
    # Verify matches were found
    assert results is not None
    assert len(results) > 0
    
    # Check if our specific match is in the results
    found_match = False
    for match in results:
        amounts = [item.amount for item in match["items"]]
        if 35000.00 in amounts:
            found_match = True
            # Verify we have items from different document types
            doc_types = set()
            for item in match["items"]:
                doc = test_session.query(Document).filter(
                    Document.doc_id == item.doc_id
                ).one()
                doc_types.add(doc.doc_type)
            
            assert len(doc_types) >= 2
    
    assert found_match


def test_detect_financial_anomalies(test_session, sample_payment_app, request):
    """Test detecting financial anomalies."""
    payment_app, _ = sample_payment_app
    test_name = request.node.name
    
    # Create a duplicate line item with slightly different description
    # but identical amount to simulate potential double-billing
    unique_item_id = f"item_dup_{test_name}_{os.getpid()}"
    duplicate_item = LineItem(
        doc_id=payment_app.doc_id,
        item_id=unique_item_id,
        item_number="5",
        description="Foundation Installation",  # Similar but not identical to "Foundation Work"
        amount=50000.00,  # Same amount as item_1
        category="construction",
        status="pending",
        metadata={}
    )
    
    test_session.add(duplicate_item)
    test_session.commit()
    
    # Run anomaly detection
    engine = FinancialAnalysisEngine(test_session)
    anomalies = engine.detect_anomalies(payment_app.doc_id)
    
    # Verify anomalies were detected
    assert anomalies is not None
    assert len(anomalies) > 0
    
    # Check if our duplicate amount anomaly was detected
    duplicate_found = False
    for anomaly in anomalies:
        if anomaly["type"] == "duplicate_amount" and anomaly["confidence"] > 0.7:
            duplicate_found = True
            break
    
    assert duplicate_found


def test_chronological_analysis(test_session, sample_payment_app, sample_change_order, request):
    """Test chronological analysis of financial transactions."""
    payment_app, _ = sample_payment_app
    change_order, _ = sample_change_order
    test_name = request.node.name
    
    # Create a chronological relationship between documents
    # The change order should modify the contract amount
    transaction1 = FinancialTransaction(
        item_id=payment_app.line_items[0].item_id,
        amount=200000.00,
        transaction_type="contract",
        status="approved",
        meta_data={"date": "2023-01-15", "description": f"Initial contract amount {test_name}_{os.getpid()}"}
    )
    
    transaction2 = FinancialTransaction(
        item_id=change_order.line_items[0].item_id,
        amount=40000.00,
        transaction_type="change",
        status="approved",
        meta_data={"date": "2023-02-10", "description": f"Change order modification {test_name}_{os.getpid()}", 
                  "reference_doc_id": payment_app.doc_id}
    )
    
    test_session.add(transaction1)
    test_session.add(transaction2)
    test_session.commit()
    
    # Run chronological analysis
    engine = FinancialAnalysisEngine(test_session)
    timeline = engine.analyze_financial_timeline("test_project")
    
    # Verify the timeline was created
    assert timeline is not None
    assert len(timeline) >= 2
    
    # Verify chronological order
    dates = [entry["date"] for entry in timeline]
    assert dates == sorted(dates)
    
    # Verify running total is updated
    running_total = 0
    for entry in timeline:
        if entry["transaction_type"] == "contract":
            running_total = entry["amount"]
        elif entry["transaction_type"] == "change":
            running_total += entry["amount"]
    
    assert running_total == 240000.00  # 200,000 + 40,000