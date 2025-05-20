"""Database tools for AI components.

This module provides specialized database query tools for AI components to
retrieve and analyze data from the database. These tools help the AI agents
perform complex queries and aggregations.
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date

logger = logging.getLogger(__name__)


def run_sql_query(db_session, query: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Run a SQL query with safety checks.
    
    Args:
        db_session: Database session
        query: SQL query to run (read-only)
        params: Optional parameters for the query
        
    Returns:
        JSON string of query results
    """
    # Security check: Ensure query is read-only
    if not _is_read_only_query(query):
        error_msg = "Security error: Only SELECT queries are allowed"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})
    
    params = params or {}
    
    try:
        # Execute query
        result = db_session.execute(query, params)
        
        # Convert to list of dictionaries
        if result.returns_rows:
            rows = []
            columns = result.keys()
            
            for row in result:
                # Convert row to dictionary with custom JSON serialization
                row_dict = {}
                for i, column in enumerate(columns):
                    value = row[i]
                    
                    # Handle special data types
                    if isinstance(value, (datetime, date)):
                        row_dict[column] = value.isoformat()
                    elif hasattr(value, "__dict__"):  # Handle ORM objects
                        row_dict[column] = str(value)
                    else:
                        row_dict[column] = value
                
                rows.append(row_dict)
            
            return json.dumps({"rows": rows, "count": len(rows)})
        else:
            return json.dumps({"message": "Query executed successfully but returned no rows"})
    
    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})


def get_document_relationships(db_session, doc_id: str) -> str:
    """Get relationships for a specific document.
    
    Args:
        db_session: Database session
        doc_id: Document ID
        
    Returns:
        JSON string of document relationships
    """
    try:
        # Query for direct relationships
        relationships_query = """
            SELECT 
                r.relationship_id,
                r.source_doc_id,
                r.target_doc_id,
                r.relationship_type,
                r.description,
                r.created_at,
                s.title AS source_title,
                s.doc_type AS source_type,
                t.title AS target_title,
                t.doc_type AS target_type
            FROM 
                document_relationships r
            JOIN
                documents s ON r.source_doc_id = s.doc_id
            JOIN
                documents t ON r.target_doc_id = t.doc_id
            WHERE 
                r.source_doc_id = :doc_id OR r.target_doc_id = :doc_id
            ORDER BY
                r.created_at DESC
        """
        
        relationships = db_session.execute(relationships_query, {"doc_id": doc_id}).fetchall()
        
        # Format the results
        formatted_relationships = []
        for rel in relationships:
            # Determine direction
            is_source = rel[1] == doc_id
            related_doc_id = rel[2] if is_source else rel[1]
            related_doc_title = rel[8] if is_source else rel[6]
            related_doc_type = rel[9] if is_source else rel[7]
            direction = "outgoing" if is_source else "incoming"
            
            formatted_relationships.append({
                "relationship_id": rel[0],
                "relationship_type": rel[3],
                "description": rel[4],
                "created_at": rel[5].isoformat() if rel[5] else None,
                "direction": direction,
                "related_doc_id": related_doc_id,
                "related_doc_title": related_doc_title,
                "related_doc_type": related_doc_type
            })
        
        # Get network depth (how many related documents)
        network_query = """
            WITH RECURSIVE document_network AS (
                -- Base case: the document itself
                SELECT 
                    doc_id, 
                    title,
                    doc_type,
                    0 AS depth
                FROM 
                    documents
                WHERE 
                    doc_id = :doc_id
                
                UNION
                
                -- Recursive case: related documents
                SELECT 
                    d.doc_id, 
                    d.title,
                    d.doc_type,
                    dn.depth + 1 AS depth
                FROM 
                    documents d
                JOIN 
                    document_relationships r ON (d.doc_id = r.target_doc_id OR d.doc_id = r.source_doc_id)
                JOIN 
                    document_network dn ON (
                        (dn.doc_id = r.source_doc_id AND d.doc_id = r.target_doc_id) OR
                        (dn.doc_id = r.target_doc_id AND d.doc_id = r.source_doc_id)
                    )
                WHERE 
                    d.doc_id != :doc_id AND
                    dn.depth < 2  -- Limit to 2 degrees of separation
            )
            
            SELECT 
                COUNT(DISTINCT doc_id) AS network_size,
                MAX(depth) AS max_depth
            FROM 
                document_network
        """
        
        network_stats = db_session.execute(network_query, {"doc_id": doc_id}).fetchone()
        
        result = {
            "doc_id": doc_id,
            "relationships": formatted_relationships,
            "relationship_count": len(formatted_relationships),
            "network_size": network_stats[0] if network_stats else 0,
            "max_depth": network_stats[1] if network_stats else 0
        }
        
        return json.dumps(result)
    
    except Exception as e:
        error_msg = f"Error getting document relationships: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})


def get_document_metadata(db_session, doc_id: str) -> str:
    """Get comprehensive metadata for a document.
    
    Args:
        db_session: Database session
        doc_id: Document ID
        
    Returns:
        JSON string of document metadata
    """
    try:
        # Query for document metadata
        doc_query = """
            SELECT 
                d.doc_id,
                d.title,
                d.doc_type,
                d.party,
                d.status,
                d.date,
                d.created_at,
                d.updated_at,
                d.original_file,
                d.file_type,
                COUNT(p.page_id) AS page_count,
                COUNT(DISTINCT l.line_item_id) AS line_item_count
            FROM 
                documents d
            LEFT JOIN
                pages p ON d.doc_id = p.doc_id
            LEFT JOIN
                line_items l ON d.doc_id = l.doc_id
            WHERE 
                d.doc_id = :doc_id
            GROUP BY
                d.doc_id
        """
        
        doc_info = db_session.execute(doc_query, {"doc_id": doc_id}).fetchone()
        
        if not doc_info:
            return json.dumps({"error": f"Document {doc_id} not found"})
        
        # Format the base document info
        result = {
            "doc_id": doc_info[0],
            "title": doc_info[1],
            "doc_type": doc_info[2],
            "party": doc_info[3],
            "status": doc_info[4],
            "date": doc_info[5].isoformat() if doc_info[5] else None,
            "created_at": doc_info[6].isoformat() if doc_info[6] else None,
            "updated_at": doc_info[7].isoformat() if doc_info[7] else None,
            "original_file": doc_info[8],
            "file_type": doc_info[9],
            "page_count": doc_info[10],
            "line_item_count": doc_info[11]
        }
        
        # Add financial summary if document has line items
        if doc_info[11] > 0:
            financial_query = """
                SELECT 
                    COUNT(line_item_id) AS item_count,
                    SUM(amount) AS total_amount,
                    MIN(amount) AS min_amount,
                    MAX(amount) AS max_amount,
                    AVG(amount) AS avg_amount
                FROM 
                    line_items
                WHERE 
                    doc_id = :doc_id
            """
            
            financial_info = db_session.execute(financial_query, {"doc_id": doc_id}).fetchone()
            
            result["financial_summary"] = {
                "item_count": financial_info[0],
                "total_amount": float(financial_info[1]) if financial_info[1] is not None else 0,
                "min_amount": float(financial_info[2]) if financial_info[2] is not None else 0,
                "max_amount": float(financial_info[3]) if financial_info[3] is not None else 0,
                "avg_amount": float(financial_info[4]) if financial_info[4] is not None else 0
            }
        
        # Add analysis flags
        flags_query = """
            SELECT 
                flag_id,
                flag_type,
                description,
                confidence,
                created_at
            FROM 
                analysis_flags
            WHERE 
                doc_id = :doc_id
            ORDER BY
                confidence DESC
        """
        
        flags = db_session.execute(flags_query, {"doc_id": doc_id}).fetchall()
        
        result["analysis_flags"] = [
            {
                "flag_id": flag[0],
                "flag_type": flag[1],
                "description": flag[2],
                "confidence": float(flag[3]) if flag[3] is not None else None,
                "created_at": flag[4].isoformat() if flag[4] else None
            }
            for flag in flags
        ]
        
        return json.dumps(result)
    
    except Exception as e:
        error_msg = f"Error getting document metadata: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})


def get_amount_references(db_session, amount: float, tolerance: float = 0.01) -> str:
    """Find all references to a specific amount across documents.
    
    Args:
        db_session: Database session
        amount: Amount to search for
        tolerance: Tolerance percentage (default 1%)
        
    Returns:
        JSON string of amount references
    """
    try:
        # Calculate tolerance range
        tolerance_value = amount * tolerance
        min_amount = amount - tolerance_value
        max_amount = amount + tolerance_value
        
        # Query for line items with this amount
        query = """
            SELECT 
                l.line_item_id,
                l.doc_id,
                l.description,
                l.amount,
                l.line_number,
                l.category,
                d.title,
                d.doc_type,
                d.party,
                d.date
            FROM 
                line_items l
            JOIN
                documents d ON l.doc_id = d.doc_id
            WHERE 
                l.amount BETWEEN :min_amount AND :max_amount
            ORDER BY
                d.date,
                l.line_number
        """
        
        items = db_session.execute(query, {
            "min_amount": min_amount,
            "max_amount": max_amount
        }).fetchall()
        
        # Format results
        references = []
        for item in items:
            references.append({
                "line_item_id": item[0],
                "doc_id": item[1],
                "description": item[2],
                "amount": float(item[3]),
                "line_number": item[4],
                "category": item[5],
                "document": {
                    "title": item[6],
                    "doc_type": item[7],
                    "party": item[8],
                    "date": item[9].isoformat() if item[9] else None
                }
            })
        
        # Get distribution by document type
        distribution_query = """
            SELECT 
                d.doc_type,
                COUNT(l.line_item_id) AS count
            FROM 
                line_items l
            JOIN
                documents d ON l.doc_id = d.doc_id
            WHERE 
                l.amount BETWEEN :min_amount AND :max_amount
            GROUP BY
                d.doc_type
            ORDER BY
                count DESC
        """
        
        distribution = db_session.execute(distribution_query, {
            "min_amount": min_amount,
            "max_amount": max_amount
        }).fetchall()
        
        # Check for suspicious patterns
        patterns_query = """
            SELECT 
                COUNT(DISTINCT doc_id) AS doc_count,
                COUNT(DISTINCT description) AS description_count,
                MIN(d.date) AS first_date,
                MAX(d.date) AS last_date
            FROM 
                line_items l
            JOIN
                documents d ON l.doc_id = d.doc_id
            WHERE 
                l.amount BETWEEN :min_amount AND :max_amount
        """
        
        patterns = db_session.execute(patterns_query, {
            "min_amount": min_amount,
            "max_amount": max_amount
        }).fetchone()
        
        # Calculate days between first and last appearance
        days_between = None
        if patterns[2] and patterns[3]:
            delta = patterns[3] - patterns[2]
            days_between = delta.days
        
        # Format final result
        result = {
            "amount": amount,
            "tolerance": tolerance,
            "min_amount": min_amount,
            "max_amount": max_amount,
            "references": references,
            "reference_count": len(references),
            "distribution": [
                {"doc_type": d[0], "count": d[1]}
                for d in distribution
            ],
            "patterns": {
                "document_count": patterns[0],
                "description_count": patterns[1],
                "first_date": patterns[2].isoformat() if patterns[2] else None,
                "last_date": patterns[3].isoformat() if patterns[3] else None,
                "days_between": days_between
            }
        }
        
        return json.dumps(result)
    
    except Exception as e:
        error_msg = f"Error getting amount references: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})


def find_date_range_activity(db_session, start_date: str, end_date: str, doc_type: Optional[str] = None,
                           party: Optional[str] = None) -> str:
    """Find activity within a date range.
    
    Args:
        db_session: Database session
        start_date: Start date (ISO format: YYYY-MM-DD)
        end_date: End date (ISO format: YYYY-MM-DD)
        doc_type: Optional document type filter
        party: Optional party filter
        
    Returns:
        JSON string of activity summary
    """
    try:
        # Validate dates
        try:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
        except ValueError:
            return json.dumps({"error": "Invalid date format. Use ISO format (YYYY-MM-DD)"})
        
        # Build query params
        params = {
            "start_date": start,
            "end_date": end
        }
        
        # Build document query
        doc_query = """
            SELECT 
                d.doc_id,
                d.title,
                d.doc_type,
                d.party,
                d.date,
                COUNT(l.line_item_id) AS item_count,
                SUM(l.amount) AS total_amount
            FROM 
                documents d
            LEFT JOIN
                line_items l ON d.doc_id = l.doc_id
            WHERE 
                d.date BETWEEN :start_date AND :end_date
        """
        
        if doc_type:
            doc_query += " AND d.doc_type = :doc_type"
            params["doc_type"] = doc_type
        
        if party:
            doc_query += " AND d.party = :party"
            params["party"] = party
        
        doc_query += """
            GROUP BY
                d.doc_id
            ORDER BY
                d.date
        """
        
        documents = db_session.execute(doc_query, params).fetchall()
        
        # Format document results
        doc_list = []
        for doc in documents:
            doc_list.append({
                "doc_id": doc[0],
                "title": doc[1],
                "doc_type": doc[2],
                "party": doc[3],
                "date": doc[4].isoformat() if doc[4] else None,
                "item_count": doc[5],
                "total_amount": float(doc[6]) if doc[6] is not None else 0
            })
        
        # Get summary statistics
        summary_query = """
            SELECT 
                COUNT(DISTINCT d.doc_id) AS doc_count,
                COUNT(DISTINCT d.doc_type) AS doc_type_count,
                COUNT(DISTINCT l.line_item_id) AS line_item_count,
                SUM(l.amount) AS total_amount
            FROM 
                documents d
            LEFT JOIN
                line_items l ON d.doc_id = l.doc_id
            WHERE 
                d.date BETWEEN :start_date AND :end_date
        """
        
        if doc_type:
            summary_query += " AND d.doc_type = :doc_type"
        
        if party:
            summary_query += " AND d.party = :party"
        
        summary = db_session.execute(summary_query, params).fetchone()
        
        # Get document type distribution
        distribution_query = """
            SELECT 
                d.doc_type,
                COUNT(DISTINCT d.doc_id) AS doc_count
            FROM 
                documents d
            WHERE 
                d.date BETWEEN :start_date AND :end_date
        """
        
        if doc_type:
            distribution_query += " AND d.doc_type = :doc_type"
        
        if party:
            distribution_query += " AND d.party = :party"
        
        distribution_query += """
            GROUP BY
                d.doc_type
            ORDER BY
                doc_count DESC
        """
        
        distribution = db_session.execute(distribution_query, params).fetchall()
        
        # Format final result
        result = {
            "date_range": {
                "start_date": start_date,
                "end_date": end_date
            },
            "filters": {
                "doc_type": doc_type,
                "party": party
            },
            "summary": {
                "document_count": summary[0],
                "document_type_count": summary[1],
                "line_item_count": summary[2],
                "total_amount": float(summary[3]) if summary[3] is not None else 0
            },
            "distribution": [
                {"doc_type": d[0], "count": d[1]}
                for d in distribution
            ],
            "documents": doc_list
        }
        
        return json.dumps(result)
    
    except Exception as e:
        error_msg = f"Error finding date range activity: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})


def get_document_changes(db_session, doc_id: str) -> str:
    """Get historical changes for a document.
    
    Args:
        db_session: Database session
        doc_id: Document ID
        
    Returns:
        JSON string of document change history
    """
    try:
        # Query for document history
        history_query = """
            SELECT 
                h.history_id,
                h.change_type,
                h.field_name,
                h.old_value,
                h.new_value,
                h.changed_at,
                u.username
            FROM 
                document_history h
            LEFT JOIN
                users u ON h.user_id = u.user_id
            WHERE 
                h.doc_id = :doc_id
            ORDER BY
                h.changed_at DESC
        """
        
        history = db_session.execute(history_query, {"doc_id": doc_id}).fetchall()
        
        # Format history
        changes = []
        for h in history:
            changes.append({
                "history_id": h[0],
                "change_type": h[1],
                "field_name": h[2],
                "old_value": h[3],
                "new_value": h[4],
                "changed_at": h[5].isoformat() if h[5] else None,
                "username": h[6]
            })
        
        # Get document revisions if they exist
        revisions_query = """
            SELECT 
                r.revision_id,
                r.revision_number,
                r.description,
                r.created_at,
                u.username
            FROM 
                document_revisions r
            LEFT JOIN
                users u ON r.user_id = u.user_id
            WHERE 
                r.doc_id = :doc_id
            ORDER BY
                r.revision_number
        """
        
        revisions = db_session.execute(revisions_query, {"doc_id": doc_id}).fetchall()
        
        # Format revisions
        formatted_revisions = []
        for r in revisions:
            formatted_revisions.append({
                "revision_id": r[0],
                "revision_number": r[1],
                "description": r[2],
                "created_at": r[3].isoformat() if r[3] else None,
                "username": r[4]
            })
        
        # Format final result
        result = {
            "doc_id": doc_id,
            "changes": changes,
            "change_count": len(changes),
            "revisions": formatted_revisions,
            "revision_count": len(formatted_revisions)
        }
        
        return json.dumps(result)
    
    except Exception as e:
        error_msg = f"Error getting document changes: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})


def get_financial_transactions(db_session, start_date: Optional[str] = None, end_date: Optional[str] = None,
                             min_amount: Optional[float] = None, max_amount: Optional[float] = None,
                             transaction_type: Optional[str] = None, limit: int = 50) -> str:
    """Get financial transactions matching criteria.
    
    Args:
        db_session: Database session
        start_date: Optional start date (ISO format: YYYY-MM-DD)
        end_date: Optional end date (ISO format: YYYY-MM-DD)
        min_amount: Optional minimum amount
        max_amount: Optional maximum amount
        transaction_type: Optional transaction type
        limit: Maximum number of results (default 50)
        
    Returns:
        JSON string of financial transactions
    """
    try:
        # Build query params
        params = {"limit": limit}
        
        # Build query
        query = """
            SELECT 
                t.transaction_id,
                t.transaction_type,
                t.amount,
                t.description,
                t.transaction_date,
                t.source_doc_id,
                t.target_doc_id,
                s.title AS source_title,
                s.doc_type AS source_doc_type,
                t2.title AS target_title,
                t2.doc_type AS target_doc_type,
                t.created_at
            FROM 
                financial_transactions t
            LEFT JOIN
                documents s ON t.source_doc_id = s.doc_id
            LEFT JOIN
                documents t2 ON t.target_doc_id = t2.doc_id
            WHERE 
                1=1
        """
        
        # Add filters
        if start_date:
            try:
                start = datetime.fromisoformat(start_date)
                query += " AND t.transaction_date >= :start_date"
                params["start_date"] = start
            except ValueError:
                return json.dumps({"error": "Invalid start_date format. Use ISO format (YYYY-MM-DD)"})
        
        if end_date:
            try:
                end = datetime.fromisoformat(end_date)
                query += " AND t.transaction_date <= :end_date"
                params["end_date"] = end
            except ValueError:
                return json.dumps({"error": "Invalid end_date format. Use ISO format (YYYY-MM-DD)"})
        
        if min_amount is not None:
            query += " AND t.amount >= :min_amount"
            params["min_amount"] = min_amount
        
        if max_amount is not None:
            query += " AND t.amount <= :max_amount"
            params["max_amount"] = max_amount
        
        if transaction_type:
            query += " AND t.transaction_type = :transaction_type"
            params["transaction_type"] = transaction_type
        
        # Add order and limit
        query += """
            ORDER BY
                t.transaction_date DESC,
                t.amount DESC
            LIMIT :limit
        """
        
        transactions = db_session.execute(query, params).fetchall()
        
        # Format results
        result = []
        for t in transactions:
            result.append({
                "transaction_id": t[0],
                "transaction_type": t[1],
                "amount": float(t[2]) if t[2] is not None else 0,
                "description": t[3],
                "transaction_date": t[4].isoformat() if t[4] else None,
                "source_doc": {
                    "doc_id": t[5],
                    "title": t[7],
                    "doc_type": t[8]
                } if t[5] else None,
                "target_doc": {
                    "doc_id": t[6],
                    "title": t[9],
                    "doc_type": t[10]
                } if t[6] else None,
                "created_at": t[11].isoformat() if t[11] else None
            })
        
        # Get summary stats
        summary_query = """
            SELECT 
                COUNT(transaction_id) AS transaction_count,
                SUM(amount) AS total_amount,
                MIN(amount) AS min_amount,
                MAX(amount) AS max_amount,
                AVG(amount) AS avg_amount
            FROM 
                financial_transactions t
            WHERE 
                1=1
        """
        
        # Add the same filters
        if start_date and "start_date" in params:
            summary_query += " AND t.transaction_date >= :start_date"
        
        if end_date and "end_date" in params:
            summary_query += " AND t.transaction_date <= :end_date"
        
        if min_amount is not None:
            summary_query += " AND t.amount >= :min_amount"
        
        if max_amount is not None:
            summary_query += " AND t.amount <= :max_amount"
        
        if transaction_type:
            summary_query += " AND t.transaction_type = :transaction_type"
        
        summary = db_session.execute(summary_query, params).fetchone()
        
        # Format final result
        return json.dumps({
            "transactions": result,
            "count": len(result),
            "summary": {
                "transaction_count": summary[0],
                "total_amount": float(summary[1]) if summary[1] is not None else 0,
                "min_amount": float(summary[2]) if summary[2] is not None else 0,
                "max_amount": float(summary[3]) if summary[3] is not None else 0,
                "avg_amount": float(summary[4]) if summary[4] is not None else 0
            },
            "filters": {
                "start_date": start_date,
                "end_date": end_date,
                "min_amount": min_amount,
                "max_amount": max_amount,
                "transaction_type": transaction_type
            }
        })
    
    except Exception as e:
        error_msg = f"Error getting financial transactions: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})


def _is_read_only_query(query: str) -> bool:
    """Check if a SQL query is read-only (SELECT only).
    
    Args:
        query: SQL query to check
        
    Returns:
        True if the query is read-only, False otherwise
    """
    # Normalize query
    normalized_query = query.strip().lower()
    
    # Remove comments
    normalized_query = re.sub(r'--.*?$', '', normalized_query, flags=re.MULTILINE)
    normalized_query = re.sub(r'/\*.*?\*/', '', normalized_query, flags=re.DOTALL)
    
    # Check for forbidden keywords that could modify data
    forbidden_keywords = [
        r'\binsert\b', r'\bupdate\b', r'\bdelete\b', r'\bdrop\b', 
        r'\balter\b', r'\bcreate\b', r'\btruncate\b', r'\bmerge\b',
        r'\bexec\b', r'\bcall\b', r'\bbegin\b', r'\bcommit\b',
        r'\brollback\b', r'\block\b', r'\bgrant\b', r'\brevoke\b'
    ]
    
    for keyword in forbidden_keywords:
        if re.search(keyword, normalized_query, re.IGNORECASE):
            return False
    
    # Check if the query starts with SELECT or WITH (for CTEs)
    # This is an additional safeguard
    is_select = re.match(r'^\s*(select|with)\b', normalized_query, re.IGNORECASE) is not None
    
    return is_select