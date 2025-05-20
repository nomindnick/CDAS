"""Tests for database tools."""

import unittest
from unittest import mock
import json
from datetime import datetime

from cdas.ai.tools.database_tools import (
    run_sql_query,
    get_document_relationships,
    get_document_metadata,
    get_amount_references,
    find_date_range_activity,
    get_document_changes,
    get_financial_transactions,
    _is_read_only_query
)


class TestDatabaseTools(unittest.TestCase):
    """Tests for the database tools module."""
    
    def setUp(self):
        """Set up test case."""
        self.db_session = mock.Mock()
        
        # Mock execute method
        self.db_session.execute.return_value.fetchall.return_value = []
        self.db_session.execute.return_value.fetchone.return_value = None
        self.db_session.execute.return_value.keys.return_value = []
    
    def test_run_sql_query_read_only(self):
        """Test running a valid read-only SQL query."""
        # Set up mock data
        rows = [
            (1, 'doc_1', 'Test Document 1'),
            (2, 'doc_2', 'Test Document 2')
        ]
        result_mock = mock.Mock()
        result_mock.returns_rows = True
        result_mock.keys.return_value = ['id', 'doc_id', 'title']
        result_mock.__iter__ = lambda _: iter(rows)
        self.db_session.execute.return_value = result_mock
        
        query = "SELECT id, doc_id, title FROM documents"
        result = run_sql_query(self.db_session, query)
        parsed_result = json.loads(result)
        
        # Check that query was executed
        self.db_session.execute.assert_called_once_with(query, {})
        
        # Check results
        self.assertEqual(len(parsed_result['rows']), 2)
        self.assertEqual(parsed_result['rows'][0]['id'], 1)
        self.assertEqual(parsed_result['rows'][0]['doc_id'], 'doc_1')
        self.assertEqual(parsed_result['rows'][0]['title'], 'Test Document 1')
        self.assertEqual(parsed_result['count'], 2)
    
    def test_run_sql_query_non_read_only(self):
        """Test rejecting non-read-only SQL queries."""
        queries = [
            "INSERT INTO documents (doc_id) VALUES ('doc_1')",
            "UPDATE documents SET title = 'New Title' WHERE doc_id = 'doc_1'",
            "DELETE FROM documents WHERE doc_id = 'doc_1'",
            "DROP TABLE documents",
            "CREATE TABLE test (id INT)",
            "ALTER TABLE documents ADD COLUMN new_col TEXT",
            "TRUNCATE TABLE documents"
        ]
        
        for query in queries:
            result = run_sql_query(self.db_session, query)
            parsed_result = json.loads(result)
            
            # Check that query was not executed
            self.db_session.execute.assert_not_called()
            
            # Check error message
            self.assertIn('error', parsed_result)
            self.assertIn('Security error', parsed_result['error'])
    
    def test_get_document_relationships(self):
        """Test getting document relationships."""
        # Mock data for direct relationships
        relationships = [
            ('rel_1', 'doc_1', 'doc_2', 'references', 'Test description', 
             datetime(2023, 1, 1), 'Source Doc', 'contract', 'Target Doc', 'invoice'),
            ('rel_2', 'doc_3', 'doc_1', 'amends', 'Another description', 
             datetime(2023, 2, 1), 'Source Doc 2', 'change_order', 'Source Doc', 'contract')
        ]
        
        # Mock data for network stats
        network_stats = (3, 2)  # network_size, max_depth
        
        # Configure mocks
        self.db_session.execute.side_effect = [
            mock.Mock(fetchall=mock.Mock(return_value=relationships)),
            mock.Mock(fetchone=mock.Mock(return_value=network_stats))
        ]
        
        result = get_document_relationships(self.db_session, 'doc_1')
        parsed_result = json.loads(result)
        
        # Check results
        self.assertEqual(parsed_result['doc_id'], 'doc_1')
        self.assertEqual(len(parsed_result['relationships']), 2)
        self.assertEqual(parsed_result['relationship_count'], 2)
        self.assertEqual(parsed_result['network_size'], 3)
        self.assertEqual(parsed_result['max_depth'], 2)
        
        # Check relationship details
        rel1 = parsed_result['relationships'][0]
        self.assertEqual(rel1['relationship_id'], 'rel_1')
        self.assertEqual(rel1['relationship_type'], 'references')
        self.assertEqual(rel1['direction'], 'outgoing')
        self.assertEqual(rel1['related_doc_id'], 'doc_2')
        
        rel2 = parsed_result['relationships'][1]
        self.assertEqual(rel2['relationship_id'], 'rel_2')
        self.assertEqual(rel2['relationship_type'], 'amends')
        self.assertEqual(rel2['direction'], 'incoming')
        self.assertEqual(rel2['related_doc_id'], 'doc_3')
    
    def test_get_document_metadata(self):
        """Test getting document metadata."""
        # Mock document info
        doc_info = (
            'doc_1', 'Test Document', 'contract', 'contractor', 'active',
            datetime(2023, 1, 1), datetime(2023, 1, 1), datetime(2023, 1, 1),
            'test.pdf', 'pdf', 10, 5
        )
        
        # Mock financial summary
        financial_info = (5, 10000.0, 1000.0, 3000.0, 2000.0)
        
        # Mock analysis flags
        flags = [
            ('flag_1', 'suspicious', 'Test flag', 0.9, datetime(2023, 1, 1)),
            ('flag_2', 'anomaly', 'Another flag', 0.8, datetime(2023, 1, 2))
        ]
        
        # Configure mocks
        self.db_session.execute.side_effect = [
            mock.Mock(fetchone=mock.Mock(return_value=doc_info)),
            mock.Mock(fetchone=mock.Mock(return_value=financial_info)),
            mock.Mock(fetchall=mock.Mock(return_value=flags))
        ]
        
        result = get_document_metadata(self.db_session, 'doc_1')
        parsed_result = json.loads(result)
        
        # Check document info
        self.assertEqual(parsed_result['doc_id'], 'doc_1')
        self.assertEqual(parsed_result['title'], 'Test Document')
        self.assertEqual(parsed_result['doc_type'], 'contract')
        self.assertEqual(parsed_result['party'], 'contractor')
        self.assertEqual(parsed_result['status'], 'active')
        self.assertEqual(parsed_result['page_count'], 10)
        self.assertEqual(parsed_result['line_item_count'], 5)
        
        # Check financial summary
        self.assertEqual(parsed_result['financial_summary']['item_count'], 5)
        self.assertEqual(parsed_result['financial_summary']['total_amount'], 10000.0)
        self.assertEqual(parsed_result['financial_summary']['min_amount'], 1000.0)
        self.assertEqual(parsed_result['financial_summary']['max_amount'], 3000.0)
        self.assertEqual(parsed_result['financial_summary']['avg_amount'], 2000.0)
        
        # Check analysis flags
        self.assertEqual(len(parsed_result['analysis_flags']), 2)
        self.assertEqual(parsed_result['analysis_flags'][0]['flag_id'], 'flag_1')
        self.assertEqual(parsed_result['analysis_flags'][0]['flag_type'], 'suspicious')
        self.assertEqual(parsed_result['analysis_flags'][0]['confidence'], 0.9)
        self.assertEqual(parsed_result['analysis_flags'][1]['flag_id'], 'flag_2')
        self.assertEqual(parsed_result['analysis_flags'][1]['flag_type'], 'anomaly')
        self.assertEqual(parsed_result['analysis_flags'][1]['confidence'], 0.8)
    
    def test_get_amount_references(self):
        """Test getting amount references."""
        # Mock line items
        items = [
            ('item_1', 'doc_1', 'Test item 1', 1000.0, 1, 'materials',
             'Test Document 1', 'contract', 'contractor', datetime(2023, 1, 1)),
            ('item_2', 'doc_2', 'Test item 2', 1000.0, 2, 'labor',
             'Test Document 2', 'invoice', 'contractor', datetime(2023, 2, 1))
        ]
        
        # Mock distribution
        distribution = [
            ('contract', 1),
            ('invoice', 1)
        ]
        
        # Mock patterns
        patterns = (2, 2, datetime(2023, 1, 1), datetime(2023, 2, 1))
        
        # Configure mocks
        self.db_session.execute.side_effect = [
            mock.Mock(fetchall=mock.Mock(return_value=items)),
            mock.Mock(fetchall=mock.Mock(return_value=distribution)),
            mock.Mock(fetchone=mock.Mock(return_value=patterns))
        ]
        
        result = get_amount_references(self.db_session, 1000.0, 0.01)
        parsed_result = json.loads(result)
        
        # Check results
        self.assertEqual(parsed_result['amount'], 1000.0)
        self.assertEqual(parsed_result['tolerance'], 0.01)
        self.assertEqual(parsed_result['min_amount'], 990.0)
        self.assertEqual(parsed_result['max_amount'], 1010.0)
        self.assertEqual(len(parsed_result['references']), 2)
        self.assertEqual(parsed_result['reference_count'], 2)
        
        # Check references
        self.assertEqual(parsed_result['references'][0]['line_item_id'], 'item_1')
        self.assertEqual(parsed_result['references'][0]['doc_id'], 'doc_1')
        self.assertEqual(parsed_result['references'][0]['amount'], 1000.0)
        self.assertEqual(parsed_result['references'][0]['document']['doc_type'], 'contract')
        
        self.assertEqual(parsed_result['references'][1]['line_item_id'], 'item_2')
        self.assertEqual(parsed_result['references'][1]['doc_id'], 'doc_2')
        self.assertEqual(parsed_result['references'][1]['amount'], 1000.0)
        self.assertEqual(parsed_result['references'][1]['document']['doc_type'], 'invoice')
        
        # Check distribution
        self.assertEqual(len(parsed_result['distribution']), 2)
        self.assertEqual(parsed_result['distribution'][0]['doc_type'], 'contract')
        self.assertEqual(parsed_result['distribution'][0]['count'], 1)
        self.assertEqual(parsed_result['distribution'][1]['doc_type'], 'invoice')
        self.assertEqual(parsed_result['distribution'][1]['count'], 1)
        
        # Check patterns
        self.assertEqual(parsed_result['patterns']['document_count'], 2)
        self.assertEqual(parsed_result['patterns']['description_count'], 2)
        self.assertEqual(parsed_result['patterns']['days_between'], 31)  # Jan 1 to Feb 1
    
    def test_find_date_range_activity(self):
        """Test finding activity within a date range."""
        # Mock document data
        documents = [
            ('doc_1', 'Test Document 1', 'contract', 'contractor',
             datetime(2023, 1, 15), 5, 10000.0),
            ('doc_2', 'Test Document 2', 'invoice', 'contractor',
             datetime(2023, 1, 20), 3, 5000.0)
        ]
        
        # Mock summary data
        summary = (2, 2, 8, 15000.0)
        
        # Mock distribution data
        distribution = [
            ('contract', 1),
            ('invoice', 1)
        ]
        
        # Configure mocks
        self.db_session.execute.side_effect = [
            mock.Mock(fetchall=mock.Mock(return_value=documents)),
            mock.Mock(fetchone=mock.Mock(return_value=summary)),
            mock.Mock(fetchall=mock.Mock(return_value=distribution))
        ]
        
        result = find_date_range_activity(
            self.db_session,
            '2023-01-01',
            '2023-01-31',
            doc_type='contract',
            party='contractor'
        )
        parsed_result = json.loads(result)
        
        # Check date range
        self.assertEqual(parsed_result['date_range']['start_date'], '2023-01-01')
        self.assertEqual(parsed_result['date_range']['end_date'], '2023-01-31')
        
        # Check filters
        self.assertEqual(parsed_result['filters']['doc_type'], 'contract')
        self.assertEqual(parsed_result['filters']['party'], 'contractor')
        
        # Check summary
        self.assertEqual(parsed_result['summary']['document_count'], 2)
        self.assertEqual(parsed_result['summary']['document_type_count'], 2)
        self.assertEqual(parsed_result['summary']['line_item_count'], 8)
        self.assertEqual(parsed_result['summary']['total_amount'], 15000.0)
        
        # Check distribution
        self.assertEqual(len(parsed_result['distribution']), 2)
        self.assertEqual(parsed_result['distribution'][0]['doc_type'], 'contract')
        self.assertEqual(parsed_result['distribution'][0]['count'], 1)
        self.assertEqual(parsed_result['distribution'][1]['doc_type'], 'invoice')
        self.assertEqual(parsed_result['distribution'][1]['count'], 1)
        
        # Check documents
        self.assertEqual(len(parsed_result['documents']), 2)
        self.assertEqual(parsed_result['documents'][0]['doc_id'], 'doc_1')
        self.assertEqual(parsed_result['documents'][0]['title'], 'Test Document 1')
        self.assertEqual(parsed_result['documents'][0]['doc_type'], 'contract')
        self.assertEqual(parsed_result['documents'][0]['party'], 'contractor')
        self.assertEqual(parsed_result['documents'][0]['item_count'], 5)
        self.assertEqual(parsed_result['documents'][0]['total_amount'], 10000.0)
    
    def test_get_document_changes(self):
        """Test getting document changes."""
        # Mock history data
        history = [
            ('history_1', 'update', 'title', 'Old Title', 'New Title',
             datetime(2023, 1, 1), 'user1'),
            ('history_2', 'update', 'status', 'draft', 'active',
             datetime(2023, 1, 2), 'user1')
        ]
        
        # Mock revisions data
        revisions = [
            ('rev_1', 1, 'Initial revision', datetime(2023, 1, 1), 'user1'),
            ('rev_2', 2, 'Updated content', datetime(2023, 1, 2), 'user1')
        ]
        
        # Configure mocks
        self.db_session.execute.side_effect = [
            mock.Mock(fetchall=mock.Mock(return_value=history)),
            mock.Mock(fetchall=mock.Mock(return_value=revisions))
        ]
        
        result = get_document_changes(self.db_session, 'doc_1')
        parsed_result = json.loads(result)
        
        # Check results
        self.assertEqual(parsed_result['doc_id'], 'doc_1')
        self.assertEqual(len(parsed_result['changes']), 2)
        self.assertEqual(parsed_result['change_count'], 2)
        self.assertEqual(len(parsed_result['revisions']), 2)
        self.assertEqual(parsed_result['revision_count'], 2)
        
        # Check changes
        self.assertEqual(parsed_result['changes'][0]['history_id'], 'history_1')
        self.assertEqual(parsed_result['changes'][0]['change_type'], 'update')
        self.assertEqual(parsed_result['changes'][0]['field_name'], 'title')
        self.assertEqual(parsed_result['changes'][0]['old_value'], 'Old Title')
        self.assertEqual(parsed_result['changes'][0]['new_value'], 'New Title')
        
        # Check revisions
        self.assertEqual(parsed_result['revisions'][0]['revision_id'], 'rev_1')
        self.assertEqual(parsed_result['revisions'][0]['revision_number'], 1)
        self.assertEqual(parsed_result['revisions'][0]['description'], 'Initial revision')
        self.assertEqual(parsed_result['revisions'][1]['revision_id'], 'rev_2')
        self.assertEqual(parsed_result['revisions'][1]['revision_number'], 2)
        self.assertEqual(parsed_result['revisions'][1]['description'], 'Updated content')
    
    def test_get_financial_transactions(self):
        """Test getting financial transactions."""
        # Mock transaction data
        transactions = [
            ('trans_1', 'payment', 1000.0, 'Test payment', datetime(2023, 1, 1),
             'doc_1', 'doc_2', 'Source Doc', 'contract', 'Target Doc', 'invoice',
             datetime(2023, 1, 1)),
            ('trans_2', 'approval', 2000.0, 'Test approval', datetime(2023, 1, 2),
             'doc_3', 'doc_4', 'Source Doc 2', 'change_order', 'Target Doc 2', 'payment_app',
             datetime(2023, 1, 2))
        ]
        
        # Mock summary data
        summary = (2, 3000.0, 1000.0, 2000.0, 1500.0)
        
        # Configure mocks
        self.db_session.execute.side_effect = [
            mock.Mock(fetchall=mock.Mock(return_value=transactions)),
            mock.Mock(fetchone=mock.Mock(return_value=summary))
        ]
        
        result = get_financial_transactions(
            self.db_session,
            start_date='2023-01-01',
            end_date='2023-01-31',
            min_amount=1000.0,
            max_amount=3000.0,
            transaction_type='payment'
        )
        parsed_result = json.loads(result)
        
        # Check results
        self.assertEqual(len(parsed_result['transactions']), 2)
        self.assertEqual(parsed_result['count'], 2)
        
        # Check transactions
        self.assertEqual(parsed_result['transactions'][0]['transaction_id'], 'trans_1')
        self.assertEqual(parsed_result['transactions'][0]['transaction_type'], 'payment')
        self.assertEqual(parsed_result['transactions'][0]['amount'], 1000.0)
        self.assertEqual(parsed_result['transactions'][0]['description'], 'Test payment')
        
        # Check summary
        self.assertEqual(parsed_result['summary']['transaction_count'], 2)
        self.assertEqual(parsed_result['summary']['total_amount'], 3000.0)
        self.assertEqual(parsed_result['summary']['min_amount'], 1000.0)
        self.assertEqual(parsed_result['summary']['max_amount'], 2000.0)
        self.assertEqual(parsed_result['summary']['avg_amount'], 1500.0)
        
        # Check filters
        self.assertEqual(parsed_result['filters']['start_date'], '2023-01-01')
        self.assertEqual(parsed_result['filters']['end_date'], '2023-01-31')
        self.assertEqual(parsed_result['filters']['min_amount'], 1000.0)
        self.assertEqual(parsed_result['filters']['max_amount'], 3000.0)
        self.assertEqual(parsed_result['filters']['transaction_type'], 'payment')
    
    def test_is_read_only_query(self):
        """Test read-only query detection."""
        # Read-only queries
        read_only_queries = [
            "SELECT * FROM documents",
            "SELECT id, title FROM documents WHERE doc_type = 'contract'",
            "WITH docs AS (SELECT * FROM documents) SELECT * FROM docs",
            "select * from documents",  # Case-insensitive
            " SELECT * FROM documents",  # Leading whitespace
            "-- Comment\nSELECT * FROM documents",  # With comment
            "/* Multi-line\ncomment */\nSELECT * FROM documents"  # Multi-line comment
        ]
        
        for query in read_only_queries:
            self.assertTrue(_is_read_only_query(query), f"Query should be read-only: {query}")
        
        # Non-read-only queries
        non_read_only_queries = [
            "INSERT INTO documents (doc_id) VALUES ('doc_1')",
            "UPDATE documents SET title = 'New Title' WHERE doc_id = 'doc_1'",
            "DELETE FROM documents WHERE doc_id = 'doc_1'",
            "DROP TABLE documents",
            "CREATE TABLE test (id INT)",
            "ALTER TABLE documents ADD COLUMN new_col TEXT",
            "TRUNCATE TABLE documents",
            "SELECT * FROM documents; DELETE FROM documents",  # Multiple statements
            "EXEC stored_procedure",
            "CALL my_procedure()",
            "BEGIN TRANSACTION",
            "GRANT SELECT ON documents TO user",
            "MERGE INTO documents USING source ON (documents.id = source.id) WHEN MATCHED THEN UPDATE SET documents.title = source.title"
        ]
        
        for query in non_read_only_queries:
            self.assertFalse(_is_read_only_query(query), f"Query should NOT be read-only: {query}")


if __name__ == '__main__':
    unittest.main()