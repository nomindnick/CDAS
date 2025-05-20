"""
Image document extractor module.
"""
import re
from typing import Dict, List, Any, Union, Optional
from pathlib import Path
import logging

import cv2
import numpy as np

from cdas.document_processor.processor import BaseExtractor
from cdas.document_processor.ocr import OCRProcessor
from cdas.document_processor.handwriting import HandwritingRecognizer

logger = logging.getLogger(__name__)


class ImageExtractor(BaseExtractor):
    """
    Extractor for image documents.
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize the image extractor.
        
        Args:
            tesseract_cmd: Path to the tesseract executable (optional)
        """
        self.ocr_processor = OCRProcessor(tesseract_cmd)
        self.handwriting_recognizer = HandwritingRecognizer(tesseract_cmd)
    
    def extract_data(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Extract data from an image document.
        
        Args:
            file_path: Path to the image document
            **kwargs: Additional extractor-specific arguments
                - detect_handwriting: Whether to detect handwritten text (default: True)
                - detect_tables: Whether to detect tables (default: True)
                - detect_form_fields: Whether to detect form fields (default: True)
            
        Returns:
            Dictionary containing extracted data
        """
        logger.info(f"Extracting data from image: {file_path}")
        
        # Convert to Path object if string
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Parse optional arguments
        detect_handwriting = kwargs.get('detect_handwriting', True)
        detect_tables = kwargs.get('detect_tables', True)
        detect_form_fields = kwargs.get('detect_form_fields', True)
        
        # Basic extracted data structure
        extracted_data = {
            "text_content": "",
            "tables": [],
            "form_fields": [],
            "handwritten_regions": [],
            "summary": {},
        }
        
        try:
            # Read the image
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError(f"Could not read image: {file_path}")
            
            # Perform OCR on the image
            ocr_result = self.ocr_processor.process_image(file_path)
            extracted_data["text_content"] = ocr_result["text"]
            
            # Add words with their positions
            extracted_data["words"] = ocr_result["words"]
            
            # Detect tables if requested
            if detect_tables:
                tables = self.ocr_processor.extract_tables_from_image(image)
                extracted_data["tables"] = tables
            
            # Detect form fields if requested
            if detect_form_fields:
                form_fields = self.ocr_processor.detect_form_fields(image)
                extracted_data["form_fields"] = form_fields
            
            # Detect handwritten text if requested
            if detect_handwriting:
                handwriting_result = self.handwriting_recognizer.recognize_text(file_path)
                extracted_data["handwritten_regions"] = handwriting_result["regions"]
                
                # Extract handwritten amounts
                handwritten_amounts = self.handwriting_recognizer.extract_handwritten_amounts(image)
                extracted_data["handwritten_amounts"] = handwritten_amounts
            
            # Add image metadata
            extracted_data["summary"] = {
                "width": image.shape[1],
                "height": image.shape[0],
                "ocr_confidence": ocr_result["confidence"],
                "table_count": len(extracted_data["tables"]),
                "form_field_count": len(extracted_data["form_fields"]),
                "handwritten_region_count": len(extracted_data.get("handwritten_regions", [])),
            }
        
        except Exception as e:
            logger.error(f"Error extracting data from image {file_path}: {str(e)}")
            raise
        
        return extracted_data
    
    def extract_line_items(self, file_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """
        Extract financial line items from an image document.
        
        Args:
            file_path: Path to the image document
            **kwargs: Additional extractor-specific arguments
            
        Returns:
            List of dictionaries, each representing a line item
        """
        logger.info(f"Extracting line items from image: {file_path}")
        
        # Extract all data first
        extracted_data = self.extract_data(file_path, **kwargs)
        
        # Initialize line items list
        line_items = []
        
        # Extract line items from tables
        if extracted_data["tables"]:
            for table_idx, table in enumerate(extracted_data["tables"]):
                table_line_items = self._extract_line_items_from_table(table, table_idx)
                line_items.extend(table_line_items)
        
        # If no line items found from tables, try text-based extraction
        if not line_items:
            text_line_items = self._extract_text_based_line_items(extracted_data["text_content"])
            line_items.extend(text_line_items)
        
        # Add handwritten amounts as line items
        if "handwritten_amounts" in extracted_data and extracted_data["handwritten_amounts"]:
            for amount_idx, amount in enumerate(extracted_data["handwritten_amounts"]):
                line_items.append({
                    "description": f"Handwritten amount {amount_idx + 1}",
                    "amount": amount["value"],
                    "quantity": None,
                    "unit_price": None,
                    "source": "handwritten",
                    "x": amount["x"],
                    "y": amount["y"],
                    "width": amount["width"],
                    "height": amount["height"],
                })
        
        return line_items
    
    def extract_metadata(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Extract metadata from an image document.
        
        Args:
            file_path: Path to the image document
            **kwargs: Additional extractor-specific arguments
            
        Returns:
            Dictionary containing document metadata
        """
        logger.info(f"Extracting metadata from image: {file_path}")
        
        # Convert to Path object if string
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Initialize metadata dictionary
        metadata = {
            "filename": file_path.name,
            "file_size": file_path.stat().st_size,
            "creation_date": None,
            "modification_date": None,
            "image_width": None,
            "image_height": None,
            "color_mode": None,
        }
        
        try:
            # Read the image
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError(f"Could not read image: {file_path}")
            
            # Extract basic image metadata
            metadata["image_width"] = image.shape[1]
            metadata["image_height"] = image.shape[0]
            metadata["color_mode"] = "color" if len(image.shape) == 3 else "grayscale"
            
            # Extract creation and modification dates from file stats
            stat = file_path.stat()
            metadata["creation_date"] = stat.st_ctime
            metadata["modification_date"] = stat.st_mtime
            
            # Perform OCR to extract document date information
            ocr_result = self.ocr_processor.process_image(file_path)
            text_content = ocr_result["text"]
            
            # Extract date information from text content
            date_patterns = {
                "document_date": [
                    r'(?:Date|DATE|Dated|DATED)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                    r'(?:Date|DATE|Dated|DATED)[:\s]+(\w+\s+\d{1,2},?\s+\d{4})',
                ],
                "effective_date": [
                    r'(?:Effective Date|EFFECTIVE DATE)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                    r'(?:Effective Date|EFFECTIVE DATE)[:\s]+(\w+\s+\d{1,2},?\s+\d{4})',
                ],
                "submission_date": [
                    r'(?:Submission Date|SUBMISSION DATE|Submitted|SUBMITTED)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                    r'(?:Submission Date|SUBMISSION DATE|Submitted|SUBMITTED)[:\s]+(\w+\s+\d{1,2},?\s+\d{4})',
                ],
            }
            
            for date_key, patterns in date_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, text_content)
                    if match:
                        metadata[date_key] = match.group(1)
                        break
        
        except Exception as e:
            logger.error(f"Error extracting metadata from image {file_path}: {str(e)}")
            raise
        
        return metadata
    
    def _extract_line_items_from_table(self, table: Dict[str, Any], table_idx: int) -> List[Dict[str, Any]]:
        """
        Extract line items from a table.
        
        Args:
            table: Table dictionary from OCR results
            table_idx: Index of the table
            
        Returns:
            List of dictionaries representing line items
        """
        line_items = []
        
        if "rows" not in table or not table["rows"]:
            return line_items
        
        # Skip tables that are too small
        if len(table["rows"]) < 2:
            return line_items
        
        # Check if this is a financial table (contains amount-like cells)
        is_financial = False
        for row in table["rows"]:
            for cell in row:
                if re.search(r'(\$\s*[\d,.]+)|(\d{1,3}(?:,\d{3})*\.\d{2})', cell):
                    is_financial = True
                    break
            if is_financial:
                break
        
        if not is_financial:
            return line_items
        
        # Try to identify column structure
        header_row = table["rows"][0] if table["rows"] else []
        
        # Find description and amount columns
        description_idx = None
        amount_idx = None
        
        for idx, header in enumerate(header_row):
            header_lower = header.lower()
            if any(kw in header_lower for kw in ["description", "item", "work"]):
                description_idx = idx
            elif any(kw in header_lower for kw in ["amount", "total", "price", "$"]):
                amount_idx = idx
        
        # If columns not identified by header, try to infer
        if description_idx is None or amount_idx is None:
            # Assume first column is description and look for amount column
            description_idx = 0
            
            # Find column with most currency values
            if amount_idx is None:
                currency_counts = [0] * len(header_row)
                for row in table["rows"][1:]:  # Skip header
                    for idx, cell in enumerate(row):
                        if re.search(r'(\$\s*[\d,.]+)|(\d{1,3}(?:,\d{3})*\.\d{2})', cell):
                            currency_counts[idx] += 1
                
                if currency_counts and max(currency_counts) > 0:
                    amount_idx = currency_counts.index(max(currency_counts))
        
        # Process rows to extract line items
        for row_idx, row in enumerate(table["rows"][1:], 1):  # Skip header row
            # Skip rows that are too short
            if len(row) <= max(description_idx or 0, amount_idx or 0):
                continue
            
            # Skip rows that appear to be subtotals/totals
            if description_idx is not None and len(row) > description_idx:
                desc_lower = row[description_idx].lower()
                if any(kw in desc_lower for kw in ["total", "subtotal", "sum"]):
                    continue
            
            # Extract line item data
            line_item = {
                "table_index": table_idx,
                "row_number": row_idx,
                "description": "",
                "amount": None,
                "quantity": None,
                "unit_price": None,
                "raw_data": row,
                "source": "table",
            }
            
            # Extract description
            if description_idx is not None and description_idx < len(row):
                line_item["description"] = row[description_idx]
            
            # Extract amount
            if amount_idx is not None and amount_idx < len(row):
                line_item["amount"] = self._extract_currency_value(row[amount_idx])
            
            # Add line item if it has meaningful data
            if line_item["description"] or line_item["amount"] is not None:
                line_items.append(line_item)
        
        return line_items
    
    def _extract_text_based_line_items(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract line items from text when table extraction fails.
        
        Args:
            text: Text content from OCR
            
        Returns:
            List of dictionaries representing line items
        """
        line_items = []
        
        # Pattern for finding line items in text format
        # Looking for patterns like "1. HVAC Installation....$5,000.00"
        line_item_pattern = re.compile(
            r'^(\d+\.?\s*)?(.*?)(\$\s*[\d,.]+|\d{1,3}(?:,\d{3})*\.\d{2})$', 
            re.MULTILINE
        )
        
        # Find all potential line items
        matches = line_item_pattern.findall(text)
        
        for i, match in enumerate(matches):
            item_num, description, amount_str = match
            
            # Clean and extract values
            description = description.strip()
            amount = self._extract_currency_value(amount_str)
            
            if description and amount is not None:
                line_items.append({
                    "row_number": i + 1,
                    "description": description,
                    "amount": amount,
                    "quantity": None,
                    "unit_price": None,
                    "raw_data": f"{item_num} {description} {amount_str}",
                    "source": "text",
                })
        
        return line_items
    
    def _extract_currency_value(self, text: str) -> Optional[float]:
        """
        Extract currency value from text.
        
        Args:
            text: Text containing a currency value
            
        Returns:
            Extracted float value or None if no valid value found
        """
        if not text:
            return None
        
        # Remove currency symbols and commas
        cleaned_text = text.replace('$', '').replace(',', '').strip()
        
        # Extract numeric value
        try:
            # Find the first sequence of digits with optional decimal point
            matches = re.search(r'-?\d+\.?\d*', cleaned_text)
            if matches:
                return float(matches.group(0))
            return None
        except (ValueError, TypeError):
            return None