"""
Helper functions for AI integration tests.

Provides common utilities for setting up test data, validating results,
and mocking AI component behavior.
"""

import os
import json
import tempfile
from typing import Dict, List, Any, Optional
from unittest import mock

from cdas.db.models import Document, Page, LineItem
from cdas.db.operations import generate_document_id


def create_test_document(session, content: str, doc_type: str = "invoice", party: str = "contractor") -> Document:
    """Create a test document in the database with specified content."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
        temp_file.write(content)
        file_path = temp_file.name
    
    # Create document in database
    doc = Document(
        doc_id=generate_document_id(),
        file_name=os.path.basename(file_path),
        file_path=file_path,
        file_hash=f"test_hash_{os.path.basename(file_path)}",
        file_size=len(content),
        file_type="txt",
        doc_type=doc_type,
        party=party,
        status="active",
        meta_data={"project_id": "test_project"}
    )
    
    # Add a page
    page = Page(
        page_number=1,
        content=content,
        meta_data={}
    )
    doc.pages.append(page)
    
    # Add to session and commit
    session.add(doc)
    session.commit()
    
    return doc


def create_test_documents_with_amounts(session, amounts_data: List[Dict[str, Any]]) -> List[Document]:
    """Create multiple test documents with specific amounts."""
    documents = []
    
    for doc_data in amounts_data:
        doc_type = doc_data.get("doc_type", "invoice")
        party = doc_data.get("party", "contractor")
        content = doc_data.get("content", "Test document content")
        amounts = doc_data.get("amounts", [])
        
        doc = create_test_document(session, content, doc_type, party)
        
        # Add line items with the specified amounts
        for idx, amount_info in enumerate(amounts):
            amount = amount_info.get("amount", 0.0)
            description = amount_info.get("description", f"Item {idx+1}")
            
            line_item = LineItem(
                item_id=f"test_item_{doc.doc_id}_{idx}",
                item_number=str(idx + 1),
                description=description,
                amount=amount,
                category=amount_info.get("category", "general"),
                status=amount_info.get("status", "pending"),
                meta_data=amount_info.get("meta_data", {})
            )
            
            doc.line_items.append(line_item)
        
        session.commit()
        documents.append(doc)
    
    return documents


def setup_mock_embedding_response(mock_client, texts: List[str], embeddings: Optional[List[List[float]]] = None):
    """Set up mock embedding responses for a list of texts."""
    if embeddings is None:
        # Generate default mock embeddings if not provided
        embeddings = []
        for i in range(len(texts)):
            # Create a unique embedding for each text (1536 dimensions)
            embedding = [(i + 1) * 0.01 * j % 1.0 for j in range(3)] * 512
            embeddings.append(embedding)
    
    # Set up the mock to return different embeddings for different texts
    def mock_create_embedding(**kwargs):
        input_text = kwargs.get("input", "")
        # Find the matching text and return its embedding
        if isinstance(input_text, list):
            input_text = input_text[0]  # Take first text if a list was provided
            
        for i, text in enumerate(texts):
            if text == input_text:
                mock_data = mock.Mock()
                mock_data.embedding = embeddings[i]
                mock_response = mock.Mock()
                mock_response.data = [mock_data]
                return mock_response
        
        # Default case if no match (should not happen in tests)
        mock_data = mock.Mock()
        mock_data.embedding = [0.1, 0.2, 0.3] * 512
        mock_response = mock.Mock()
        mock_response.data = [mock_data]
        return mock_response
    
    mock_client.embeddings.create.side_effect = mock_create_embedding


def setup_mock_completion_response(mock_client, responses: Dict[str, str]):
    """Set up mock completion responses for different prompts."""
    def mock_create_completion(**kwargs):
        prompt = kwargs.get("prompt", "")
        messages = kwargs.get("messages", [])
        
        # Get the prompt from messages if available
        if messages and isinstance(messages, list):
            for message in messages:
                if message.get("role") == "user":
                    prompt = message.get("content", "")
                    break
        
        # Find a matching response
        matching_response = "Default mock response"
        for key, value in responses.items():
            if key in prompt:
                matching_response = value
                break
        
        # Create mock response
        mock_message = mock.Mock()
        mock_message.content = matching_response
        
        # For OpenAI format
        mock_choice = mock.Mock()
        mock_choice.message = mock_message
        
        mock_completion = mock.Mock()
        mock_completion.choices = [mock_choice]
        
        return mock_completion
    
    mock_client.chat.completions.create.side_effect = mock_create_completion


def setup_mock_anthropic_response(mock_client, responses: Dict[str, str]):
    """Set up mock Anthropic responses for different prompts."""
    def mock_create_message(**kwargs):
        prompt = kwargs.get("prompt", "")
        messages = kwargs.get("messages", [])
        
        # Get the prompt from messages if available
        if messages and isinstance(messages, list):
            for message in messages:
                if message.get("role") == "user":
                    prompt = message.get("content", "")
                    break
        
        # Find a matching response
        matching_response = "Default mock response"
        for key, value in responses.items():
            if key in prompt:
                matching_response = value
                break
        
        # Create mock response for Anthropic format
        mock_content = mock.Mock()
        mock_content.text = matching_response
        
        mock_message = mock.Mock()
        mock_message.content = [mock_content]
        
        return mock_message
    
    mock_client.messages.create.side_effect = mock_create_message


def verify_vector_similarity(vector1: List[float], vector2: List[float], min_similarity: float = 0.5) -> bool:
    """Verify similarity between two vectors using cosine similarity."""
    from numpy import dot
    from numpy.linalg import norm
    
    # Calculate cosine similarity
    similarity = dot(vector1, vector2) / (norm(vector1) * norm(vector2))
    
    return similarity >= min_similarity


def verify_report_structure(report_content: str, required_elements: List[str]) -> bool:
    """Verify that a report contains required structural elements."""
    for element in required_elements:
        if element.lower() not in report_content.lower():
            return False
    return True


def verify_evidence_chain(evidence_chain: str, amount: float, doc_ids: List[str]) -> bool:
    """Verify that an evidence chain contains an amount and document references."""
    # Check for amount (with some formatting variations)
    amount_str = str(amount)
    amount_formatted = f"${amount:,.2f}"
    
    has_amount = amount_str in evidence_chain or amount_formatted in evidence_chain
    
    # Check for document IDs
    has_all_docs = True
    for doc_id in doc_ids:
        if doc_id not in evidence_chain:
            has_all_docs = False
            break
    
    return has_amount and has_all_docs