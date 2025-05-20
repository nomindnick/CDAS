"""Text vectorization for semantic search.

This module provides functionality for preprocessing and vectorizing text
for semantic search operations, including chunking, cleaning, and optimizing
text before embedding generation.
"""

import logging
import re
import unicodedata
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import numpy as np

logger = logging.getLogger(__name__)


class Vectorizer:
    """Manages text preprocessing and vectorization for semantic search."""
    
    def __init__(self, embedding_manager, config: Optional[Dict[str, Any]] = None):
        """Initialize the vectorizer.
        
        Args:
            embedding_manager: Instance of EmbeddingManager for generating embeddings
            config: Optional configuration dictionary
        """
        self.embedding_manager = embedding_manager
        self.config = config or {}
        
        # Chunking settings
        self.chunk_size = self.config.get('chunk_size', 1000)  # Characters per chunk
        self.chunk_overlap = self.config.get('chunk_overlap', 200)  # Overlap between chunks
        
        # Preprocessing settings
        self.min_chunk_length = self.config.get('min_chunk_length', 50)  # Minimum chunk length
        self.remove_extra_whitespace = self.config.get('remove_extra_whitespace', True)
        self.normalize_unicode = self.config.get('normalize_unicode', True)
        self.lowercase = self.config.get('lowercase', False)  # Default False to preserve meaning
        
        # Document structure recognition
        self.preserve_structure = self.config.get('preserve_structure', True)
        self.structure_markers = self.config.get('structure_markers', [
            # Headers and sections
            {'pattern': r'^#+\s+(.+)$', 'weight': 1.5},
            {'pattern': r'^Title:(.+)$', 'weight': 1.5},
            
            # Numbered lists and bullets
            {'pattern': r'^\d+\.\s+(.+)$', 'weight': 1.2},
            {'pattern': r'^\*\s+(.+)$', 'weight': 1.2},
            {'pattern': r'^-\s+(.+)$', 'weight': 1.2},
            
            # Tables (simplified pattern)
            {'pattern': r'^\|(.+)\|$', 'weight': 1.3},
            
            # Dates
            {'pattern': r'\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}', 'weight': 1.4},
            
            # Currency amounts
            {'pattern': r'\$\d{1,3}(,\d{3})*(\.\d{2})?', 'weight': 1.8},
            
            # Percentages
            {'pattern': r'\d+(\.\d+)?%', 'weight': 1.5}
        ])
        
        # Special handlers
        self._register_special_handlers()
    
    def vectorize_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Vectorize text into chunks with embeddings.
        
        Args:
            text: Text to vectorize
            metadata: Optional metadata to associate with vectors
            
        Returns:
            List of dictionaries with chunk text, embedding, and metadata
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for vectorization")
            return []
        
        try:
            # Preprocess the text
            processed_text = self._preprocess_text(text)
            
            # Split into chunks
            chunks = self._chunk_text(processed_text)
            
            # Filter out chunks that are too small
            chunks = [chunk for chunk in chunks if len(chunk) >= self.min_chunk_length]
            
            if not chunks:
                logger.warning("No valid chunks after processing")
                return []
            
            # Generate embeddings for each chunk
            result = []
            
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embedding_manager.generate_embeddings(chunk)
                
                # Create result with metadata
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_length': len(chunk),
                    'vectorizer_version': '1.0'
                })
                
                result.append({
                    'text': chunk,
                    'embedding': embedding,
                    'metadata': chunk_metadata
                })
            
            logger.info(f"Vectorized text into {len(result)} chunks")
            return result
        
        except Exception as e:
            logger.error(f"Error vectorizing text: {str(e)}")
            return []
    
    def vectorize_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Vectorize a document dictionary.
        
        Args:
            document: Document dictionary with text and metadata
            
        Returns:
            List of dictionaries with chunk text, embedding, and metadata
        """
        if 'content' not in document:
            logger.error("Document missing 'content' field")
            return []
        
        try:
            text = document['content']
            
            # Build metadata from document
            metadata = {k: v for k, v in document.items() if k != 'content'}
            
            # Add document type specific processing
            doc_type = document.get('doc_type', 'unknown')
            if doc_type in self.special_handlers:
                handler = self.special_handlers[doc_type]
                logger.debug(f"Using special handler for document type: {doc_type}")
                return handler(text, metadata)
            else:
                # Use standard processing
                return self.vectorize_text(text, metadata)
        
        except Exception as e:
            logger.error(f"Error vectorizing document: {str(e)}")
            return []
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before chunking.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Apply unicode normalization if configured
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespace if configured
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # Remove leading whitespace
            text = re.sub(r'\s+$', '', text, flags=re.MULTILINE)  # Remove trailing whitespace
        
        # Apply lowercase if configured
        if self.lowercase:
            text = text.lower()
        
        return text
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Check if we should use structure-aware chunking
        if self.preserve_structure:
            return self._structure_aware_chunking(text)
        else:
            return self._simple_chunking(text)
    
    def _simple_chunking(self, text: str) -> List[str]:
        """Split text into chunks based on size only.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # If text is shorter than chunk size, return it as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # Otherwise, create overlapping chunks
        start = 0
        while start < len(text):
            # Extract chunk
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # If we're not at the start of the text, try to find a good split point
            if start > 0:
                # Look for sentence boundaries, paragraphs, or spaces to break on
                # Prioritize paragraph breaks, then sentence breaks, then word breaks
                paragraph_break = chunk.find('\n\n')
                sentence_break = -1
                for pattern in ['. ', '? ', '! ', '.\n', '?\n', '!\n']:
                    pos = chunk.find(pattern)
                    if pos > 0 and (sentence_break == -1 or pos < sentence_break):
                        sentence_break = pos + len(pattern)
                
                word_break = chunk.rfind(' ', 0, min(100, len(chunk)))
                
                # Use the most appropriate break point
                if paragraph_break > 100:  # Avoid tiny chunks
                    chunk = chunk[:paragraph_break].strip()
                elif sentence_break > 50:  # Avoid tiny chunks
                    chunk = chunk[:sentence_break].strip()
                elif word_break > 0:
                    chunk = chunk[:word_break].strip()
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            if start + len(chunk) >= len(text):
                break
            
            start = start + len(chunk) - self.chunk_overlap
        
        return chunks
    
    def _structure_aware_chunking(self, text: str) -> List[str]:
        """Split text into chunks while preserving document structure.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
            
            # Check if paragraph fits in current chunk
            if current_length + len(paragraph) + 1 <= self.chunk_size:
                # Add to current chunk
                current_chunk.append(paragraph)
                current_length += len(paragraph) + 1  # +1 for newline
            else:
                # Check if we have anything in the current chunk
                if current_chunk:
                    # Finalize current chunk
                    chunks.append('\n\n'.join(current_chunk))
                
                # Start a new chunk
                if len(paragraph) > self.chunk_size:
                    # If paragraph is too big, use simple chunking for it
                    paragraph_chunks = self._simple_chunking(paragraph)
                    for p_chunk in paragraph_chunks:
                        chunks.append(p_chunk)
                    
                    # Reset current chunk
                    current_chunk = []
                    current_length = 0
                else:
                    # Start new chunk with this paragraph
                    current_chunk = [paragraph]
                    current_length = len(paragraph)
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        # If we need to add more context, create overlapping chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapping_chunks = []
            
            for i, chunk in enumerate(chunks):
                if i > 0:
                    # Get a portion of the previous chunk
                    prev_chunk = chunks[i-1]
                    overlap_size = min(self.chunk_overlap, len(prev_chunk))
                    overlap_text = prev_chunk[-overlap_size:] if overlap_size > 0 else ""
                    
                    # Add overlap text to current chunk if it doesn't exceed max size
                    if len(overlap_text) + len(chunk) <= self.chunk_size * 1.1:  # Allow slight exceeding
                        overlapping_chunks.append(overlap_text + "\n\n" + chunk)
                    else:
                        overlapping_chunks.append(chunk)
                else:
                    overlapping_chunks.append(chunk)
            
            return overlapping_chunks
        
        return chunks
    
    def _detect_important_content(self, text: str) -> List[Dict[str, Any]]:
        """Detect important parts of text based on structure markers.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of dictionaries with important text and weight
        """
        if not text or not self.structure_markers:
            return []
        
        important_parts = []
        
        # Apply each structure marker pattern
        for marker in self.structure_markers:
            pattern = marker['pattern']
            weight = marker['weight']
            
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                full_match = match.group(0)
                
                # If the pattern has a capture group, use that as the important text
                if match.lastindex and match.lastindex >= 1:
                    important_text = match.group(1).strip()
                else:
                    important_text = full_match.strip()
                
                # Add to list of important parts
                important_parts.append({
                    'text': important_text,
                    'weight': weight,
                    'start': match.start(),
                    'end': match.end()
                })
        
        return important_parts
    
    def _register_special_handlers(self) -> None:
        """Register special handlers for different document types."""
        self.special_handlers = {
            'invoice': self._handle_invoice,
            'payment_app': self._handle_payment_application,
            'change_order': self._handle_change_order,
            'contract': self._handle_contract,
            'schedule': self._handle_schedule,
            'correspondence': self._handle_correspondence
        }
    
    def _handle_invoice(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Special handler for invoice documents.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of vectors
        """
        # Extract invoice-specific information
        invoice_number = self._extract_pattern(text, r'Invoice\s*(?:#|No|Number)[:\s]*([A-Za-z0-9\-]+)', 'unknown')
        invoice_date = self._extract_pattern(text, r'(?:Date|Invoice\s*Date)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})', 'unknown')
        total_amount = self._extract_pattern(text, r'(?:Total|Amount\s*Due|Total\s*Amount)[:\s]*\$?([\d,]+\.\d{2})', 'unknown')
        
        # Update metadata with invoice-specific information
        metadata['invoice_number'] = invoice_number
        metadata['invoice_date'] = invoice_date
        metadata['total_amount'] = total_amount
        
        # Use standard processing with enhanced metadata
        return self.vectorize_text(text, metadata)
    
    def _handle_payment_application(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Special handler for payment application documents.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of vectors
        """
        # Extract payment application-specific information
        app_number = self._extract_pattern(text, r'(?:Application|Pay\s*App|Payment\s*Application)\s*(?:#|No|Number)[:\s]*(\d+)', 'unknown')
        app_date = self._extract_pattern(text, r'(?:Date|Application\s*Date)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})', 'unknown')
        current_payment = self._extract_pattern(text, r'(?:Current\s*Payment|This\s*Payment)[:\s]*\$?([\d,]+\.\d{2})', 'unknown')
        contract_sum = self._extract_pattern(text, r'(?:Contract\s*Sum|Original\s*Contract\s*Sum)[:\s]*\$?([\d,]+\.\d{2})', 'unknown')
        
        # Update metadata with payment app-specific information
        metadata['application_number'] = app_number
        metadata['application_date'] = app_date
        metadata['current_payment'] = current_payment
        metadata['contract_sum'] = contract_sum
        
        # Use standard processing with enhanced metadata
        return self.vectorize_text(text, metadata)
    
    def _handle_change_order(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Special handler for change order documents.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of vectors
        """
        # Extract change order-specific information
        co_number = self._extract_pattern(text, r'(?:Change\s*Order|CO)\s*(?:#|No|Number)[:\s]*([A-Za-z0-9\-]+)', 'unknown')
        co_date = self._extract_pattern(text, r'(?:Date|CO\s*Date)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})', 'unknown')
        amount = self._extract_pattern(text, r'(?:Amount|CO\s*Amount|Change\s*Order\s*Amount)[:\s]*\$?([\d,]+\.\d{2})', 'unknown')
        status = self._extract_pattern(text, r'(?:Status)[:\s]*(\w+)', 'unknown')
        
        # Update metadata with change order-specific information
        metadata['change_order_number'] = co_number
        metadata['change_order_date'] = co_date
        metadata['amount'] = amount
        metadata['status'] = status
        
        # Use standard processing with enhanced metadata
        return self.vectorize_text(text, metadata)
    
    def _handle_contract(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Special handler for contract documents.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of vectors
        """
        # For contracts, we want smaller chunks with more overlap
        original_chunk_size = self.chunk_size
        original_chunk_overlap = self.chunk_overlap
        
        # Temporarily modify chunking parameters
        self.chunk_size = min(800, self.chunk_size)
        self.chunk_overlap = max(150, self.chunk_overlap)
        
        try:
            # Extract contract-specific information
            contract_date = self._extract_pattern(text, r'(?:Date|Contract\s*Date)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})', 'unknown')
            contract_sum = self._extract_pattern(text, r'(?:Contract\s*Sum|Contract\s*Amount)[:\s]*\$?([\d,]+\.\d{2})', 'unknown')
            
            # Update metadata with contract-specific information
            metadata['contract_date'] = contract_date
            metadata['contract_sum'] = contract_sum
            
            # Use standard processing with enhanced metadata
            return self.vectorize_text(text, metadata)
        
        finally:
            # Restore original chunking parameters
            self.chunk_size = original_chunk_size
            self.chunk_overlap = original_chunk_overlap
    
    def _handle_schedule(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Special handler for schedule documents.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of vectors
        """
        # Extract schedule-specific information
        schedule_date = self._extract_pattern(text, r'(?:Date|Schedule\s*Date)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})', 'unknown')
        revision = self._extract_pattern(text, r'(?:Revision|Rev)[:\s]*([A-Za-z0-9\-]+)', 'unknown')
        
        # Update metadata with schedule-specific information
        metadata['schedule_date'] = schedule_date
        metadata['revision'] = revision
        
        # Use standard processing with enhanced metadata
        return self.vectorize_text(text, metadata)
    
    def _handle_correspondence(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Special handler for correspondence documents.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of vectors
        """
        # Extract correspondence-specific information
        date = self._extract_pattern(text, r'(?:Date)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})', 'unknown')
        subject = self._extract_pattern(text, r'(?:Subject|Re|Regarding)[:\s]*([^\n]+)', 'unknown')
        from_party = self._extract_pattern(text, r'(?:From)[:\s]*([^\n]+)', 'unknown')
        to_party = self._extract_pattern(text, r'(?:To)[:\s]*([^\n]+)', 'unknown')
        
        # Update metadata with correspondence-specific information
        metadata['date'] = date
        metadata['subject'] = subject
        metadata['from'] = from_party
        metadata['to'] = to_party
        
        # Use standard processing with enhanced metadata
        return self.vectorize_text(text, metadata)
    
    def _extract_pattern(self, text: str, pattern: str, default: str = '') -> str:
        """Extract a pattern from text.
        
        Args:
            text: Text to search
            pattern: Regex pattern with a capture group
            default: Default value if not found
            
        Returns:
            Extracted text or default
        """
        match = re.search(pattern, text)
        if match and match.group(1):
            return match.group(1).strip()
        
        return default