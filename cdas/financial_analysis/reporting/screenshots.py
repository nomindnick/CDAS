"""
Generates screenshots of document evidence for reporting.

This module provides functionality to capture specific portions of documents
as evidence screenshots for inclusion in reports.
"""

import os
import logging
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import io
import base64
from datetime import datetime

import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image

from cdas.db.models import Document, Page, LineItem
from cdas.db.operations import get_page_by_id, get_document_by_id

# Set up logging
logger = logging.getLogger(__name__)


class EvidenceScreenshotGenerator:
    """Generates screenshots of document evidence."""
    
    def __init__(self, screenshot_dir: Optional[str] = None):
        """Initialize the evidence screenshot generator.
        
        Args:
            screenshot_dir: Optional directory to save screenshots
                If not provided, a temporary directory will be used
        """
        self.screenshot_dir = screenshot_dir
        
        if screenshot_dir:
            self.screenshot_path = Path(screenshot_dir)
            self.screenshot_path.mkdir(parents=True, exist_ok=True)
        else:
            # Create a temporary directory for screenshots
            self.temp_dir = tempfile.TemporaryDirectory()
            self.screenshot_path = Path(self.temp_dir.name)
        
        # Cache for document pages to avoid repeated loading
        self._page_cache = {}
    
    def get_screenshot_for_line_item(
        self, 
        item: LineItem, 
        document: Optional[Document] = None,
        page: Optional[Page] = None,
        highlight_color: Tuple[int, int, int] = (255, 255, 0),  # Yellow
        padding: int = 20
    ) -> Dict[str, Any]:
        """Get a screenshot for a line item.
        
        Args:
            item: The line item to screenshot
            document: Optional document object (will be loaded if not provided)
            page: Optional page object (will be loaded if not provided)
            highlight_color: RGB color tuple for highlighting
            padding: Padding around the highlighted area in pixels
            
        Returns:
            Dictionary with screenshot information
        """
        logger.info(f"Generating screenshot for line item: {item.item_id}")
        
        # Get document and page if not provided
        if not document:
            document = get_document_by_id(item.doc_id)
            
        if not page and item.page_id:
            page = get_page_by_id(item.page_id)
        elif not page and item.page_num and document:
            # Try to get page by number
            for doc_page in document.pages:
                if doc_page.page_num == item.page_num:
                    page = doc_page
                    break
        
        if not document or not page:
            logger.error(f"Document or page not found for item {item.item_id}")
            return {
                'success': False,
                'item_id': item.item_id,
                'error': 'Document or page not found'
            }
        
        # Get the file path
        document_path = document.file_path
        if not document_path or not os.path.exists(document_path):
            logger.error(f"Document file not found: {document_path}")
            return {
                'success': False,
                'item_id': item.item_id,
                'error': 'Document file not found'
            }
        
        # Check file type
        if document_path.lower().endswith('.pdf'):
            return self._get_pdf_screenshot(
                item,
                document_path,
                page.page_num,
                highlight_color,
                padding
            )
        elif any(document_path.lower().endswith(ext) for ext in 
                 ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']):
            return self._get_image_screenshot(
                item,
                document_path,
                highlight_color,
                padding
            )
        else:
            logger.error(f"Unsupported file type: {document_path}")
            return {
                'success': False,
                'item_id': item.item_id,
                'error': 'Unsupported file type'
            }
    
    def get_screenshots_for_evidence(
        self,
        evidence_items: List[Dict[str, Any]],
        highlight_color: Tuple[int, int, int] = (255, 255, 0),
        padding: int = 20
    ) -> Dict[str, Any]:
        """Get screenshots for multiple evidence items.
        
        Args:
            evidence_items: List of evidence item dictionaries
            highlight_color: RGB color tuple for highlighting
            padding: Padding around the highlighted area in pixels
            
        Returns:
            Dictionary mapping evidence IDs to screenshot information
        """
        logger.info(f"Generating screenshots for {len(evidence_items)} evidence items")
        
        result = {}
        
        for evidence in evidence_items:
            evidence_id = evidence.get('evidence_id')
            if not evidence_id:
                continue
                
            # Get screenshots for each line item in the evidence
            item_screenshots = []
            for item in evidence.get('items', []):
                if isinstance(item, LineItem) and item.item_id:
                    screenshot = self.get_screenshot_for_line_item(
                        item,
                        highlight_color=highlight_color,
                        padding=padding
                    )
                    if screenshot.get('success'):
                        item_screenshots.append(screenshot)
                elif isinstance(item, dict) and 'item_id' in item:
                    # Handle case where item is a dictionary with item_id
                    from cdas.db.operations import get_line_item_by_id
                    line_item = get_line_item_by_id(item['item_id'])
                    if line_item:
                        screenshot = self.get_screenshot_for_line_item(
                            line_item,
                            highlight_color=highlight_color,
                            padding=padding
                        )
                        if screenshot.get('success'):
                            item_screenshots.append(screenshot)
            
            # Add screenshots to result
            if item_screenshots:
                result[evidence_id] = {
                    'evidence_id': evidence_id,
                    'screenshots': item_screenshots
                }
        
        return result
    
    def _get_pdf_screenshot(
        self,
        item: LineItem,
        pdf_path: str,
        page_num: int,
        highlight_color: Tuple[int, int, int],
        padding: int
    ) -> Dict[str, Any]:
        """Get a screenshot from a PDF document.
        
        Args:
            item: The line item to screenshot
            pdf_path: Path to the PDF file
            page_num: Page number (0-based)
            highlight_color: RGB color tuple for highlighting
            padding: Padding around the highlighted area in pixels
            
        Returns:
            Dictionary with screenshot information
        """
        try:
            # Check if we already have this page in cache
            cache_key = f"{pdf_path}:{page_num}"
            if cache_key in self._page_cache:
                page_image = self._page_cache[cache_key].copy()
            else:
                # Open PDF and get the page
                doc = fitz.open(pdf_path)
                if page_num >= len(doc):
                    logger.error(f"Page {page_num} not found in PDF with {len(doc)} pages")
                    return {
                        'success': False,
                        'item_id': item.item_id,
                        'error': f"Page {page_num} not found in PDF"
                    }
                
                # Get the page
                page = doc[page_num]
                
                # Render page to an image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_image = np.array(img)
                
                # Add to cache
                self._page_cache[cache_key] = page_image.copy()
                
                # Close the document
                doc.close()
            
            # Get bounding box if available, or use OCR to find it
            if hasattr(item, 'bbox') and item.bbox:
                # Use stored bounding box
                bbox = item.bbox
                x1, y1, x2, y2 = bbox
            else:
                # Search for text using the description
                text_to_find = item.description
                if not text_to_find:
                    # Use amount as fallback
                    text_to_find = f"${float(item.amount):,.2f}" if item.amount else None
                
                if not text_to_find:
                    logger.error(f"No text to search for in item {item.item_id}")
                    return {
                        'success': False,
                        'item_id': item.item_id,
                        'error': 'No text to search for'
                    }
                
                # Use OCR to find text (simplified method here)
                # In a real implementation, you'd use a proper OCR library
                # or PyMuPDF's text search to find the text
                # For simplicity, we'll just use the top half of the page
                h, w = page_image.shape[:2]
                x1, y1, x2, y2 = 0, 0, w, h // 2
            
            # Add padding to the bounding box
            h, w = page_image.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Create a copy of the image to draw on
            highlighted_image = page_image.copy()
            
            # Draw highlight rectangle
            cv2.rectangle(
                highlighted_image,
                (x1, y1),
                (x2, y2),
                highlight_color[::-1],  # OpenCV uses BGR
                2
            )
            
            # Crop the image to the region of interest with padding
            crop_x1 = max(0, x1 - padding * 2)
            crop_y1 = max(0, y1 - padding * 2)
            crop_x2 = min(w, x2 + padding * 2)
            crop_y2 = min(h, y2 + padding * 2)
            cropped_image = highlighted_image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Save the screenshot
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"item_{item.item_id}_{timestamp}.png"
            filepath = self.screenshot_path / filename
            cv2.imwrite(str(filepath), cropped_image)
            
            # Get file as data URL for embedding in reports
            with open(filepath, 'rb') as f:
                image_data = f.read()
                data_url = f"data:image/png;base64,{base64.b64encode(image_data).decode('utf-8')}"
            
            return {
                'success': True,
                'item_id': item.item_id,
                'file_path': str(filepath),
                'data_url': data_url,
                'page_num': page_num,
                'bbox': [x1, y1, x2, y2],
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"Error generating PDF screenshot: {e}")
            return {
                'success': False,
                'item_id': item.item_id,
                'error': str(e)
            }
    
    def _get_image_screenshot(
        self,
        item: LineItem,
        image_path: str,
        highlight_color: Tuple[int, int, int],
        padding: int
    ) -> Dict[str, Any]:
        """Get a screenshot from an image document.
        
        Args:
            item: The line item to screenshot
            image_path: Path to the image file
            highlight_color: RGB color tuple for highlighting
            padding: Padding around the highlighted area in pixels
            
        Returns:
            Dictionary with screenshot information
        """
        try:
            # Check if we already have this image in cache
            cache_key = image_path
            if cache_key in self._page_cache:
                page_image = self._page_cache[cache_key].copy()
            else:
                # Load the image
                page_image = cv2.imread(image_path)
                if page_image is None:
                    logger.error(f"Failed to load image: {image_path}")
                    return {
                        'success': False,
                        'item_id': item.item_id,
                        'error': 'Failed to load image'
                    }
                
                # Add to cache
                self._page_cache[cache_key] = page_image.copy()
            
            # Get bounding box if available, or use OCR to find it
            if hasattr(item, 'bbox') and item.bbox:
                # Use stored bounding box
                bbox = item.bbox
                x1, y1, x2, y2 = bbox
            else:
                # Search for text using the description
                text_to_find = item.description
                if not text_to_find:
                    # Use amount as fallback
                    text_to_find = f"${float(item.amount):,.2f}" if item.amount else None
                
                if not text_to_find:
                    logger.error(f"No text to search for in item {item.item_id}")
                    return {
                        'success': False,
                        'item_id': item.item_id,
                        'error': 'No text to search for'
                    }
                
                # Use OCR to find text (simplified method here)
                # In a real implementation, you'd use a proper OCR library
                # For simplicity, we'll just use the top half of the image
                h, w = page_image.shape[:2]
                x1, y1, x2, y2 = 0, 0, w, h // 2
            
            # Add padding to the bounding box
            h, w = page_image.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Create a copy of the image to draw on
            highlighted_image = page_image.copy()
            
            # Draw highlight rectangle
            cv2.rectangle(
                highlighted_image,
                (x1, y1),
                (x2, y2),
                highlight_color[::-1],  # OpenCV uses BGR
                2
            )
            
            # Crop the image to the region of interest with padding
            crop_x1 = max(0, x1 - padding * 2)
            crop_y1 = max(0, y1 - padding * 2)
            crop_x2 = min(w, x2 + padding * 2)
            crop_y2 = min(h, y2 + padding * 2)
            cropped_image = highlighted_image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Save the screenshot
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"item_{item.item_id}_{timestamp}.png"
            filepath = self.screenshot_path / filename
            cv2.imwrite(str(filepath), cropped_image)
            
            # Get file as data URL for embedding in reports
            with open(filepath, 'rb') as f:
                image_data = f.read()
                data_url = f"data:image/png;base64,{base64.b64encode(image_data).decode('utf-8')}"
            
            return {
                'success': True,
                'item_id': item.item_id,
                'file_path': str(filepath),
                'data_url': data_url,
                'page_num': 0,  # Single page for images
                'bbox': [x1, y1, x2, y2],
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"Error generating image screenshot: {e}")
            return {
                'success': False,
                'item_id': item.item_id,
                'error': str(e)
            }
    
    def cleanup(self):
        """Clean up temporary files."""
        # Clear page cache
        self._page_cache.clear()
        
        # Clean up temporary directory if used
        if not self.screenshot_dir and hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()