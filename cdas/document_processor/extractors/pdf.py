"""
PDF document extractor module.
"""
import re
from typing import Dict, List, Any, Union, Optional
from pathlib import Path
import logging

import pdfplumber
from pdfplumber.page import Page

from cdas.document_processor.processor import BaseExtractor

logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    """
    Extractor for PDF documents.
    """
    
    def __init__(self):
        """Initialize the PDF extractor."""
        pass
    
    def extract_data(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Extract data from a PDF document.
        
        Args:
            file_path: Path to the PDF document
            **kwargs: Additional extractor-specific arguments
            
        Returns:
            Dictionary containing extracted data
        """
        logger.info(f"Extracting data from PDF: {file_path}")
        
        # Convert to Path object if string
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Basic extracted data structure
        extracted_data = {
            "text_content": "",
            "pages": [],
            "tables": [],
            "summary": {},
        }
        
        try:
            with pdfplumber.open(file_path) as pdf:
                # Extract text from each page
                for i, page in enumerate(pdf.pages):
                    page_data = self._extract_page_data(page, page_number=i+1)
                    extracted_data["pages"].append(page_data)
                    extracted_data["text_content"] += page_data["text"] + "\n\n"
                
                # Extract document-level data
                extracted_data["summary"] = {
                    "page_count": len(pdf.pages),
                    "document_info": pdf.metadata,
                }
                
                # Process all tables across pages and deduplicate
                all_tables = []
                for page_data in extracted_data["pages"]:
                    all_tables.extend(page_data["tables"])
                
                # Store only unique tables based on their content signature
                unique_tables = []
                table_signatures = set()
                for table in all_tables:
                    signature = self._get_table_signature(table["data"])
                    if signature not in table_signatures:
                        table_signatures.add(signature)
                        unique_tables.append(table)
                
                extracted_data["tables"] = unique_tables
        
        except Exception as e:
            logger.error(f"Error extracting data from PDF {file_path}: {str(e)}")
            raise
        
        return extracted_data
    
    def extract_line_items(self, file_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """
        Extract financial line items from a PDF document.
        
        Args:
            file_path: Path to the PDF document
            **kwargs: Additional extractor-specific arguments
            
        Returns:
            List of dictionaries, each representing a line item
        """
        logger.info(f"Extracting line items from PDF: {file_path}")
        
        # Extract all data first
        extracted_data = self.extract_data(file_path, **kwargs)
        
        # Initialize line items list
        line_items = []
        
        # Process tables to find financial line items
        for table in extracted_data["tables"]:
            table_data = table["data"]
            
            # Skip small tables or likely headers only
            if len(table_data) <= 1 or len(table_data[0]) <= 1:
                continue
            
            # Try to identify if this is a financial table
            financial_table = self._is_financial_table(table_data)
            if financial_table:
                # Extract line items from the financial table
                table_line_items = self._extract_financial_line_items(table_data)
                line_items.extend(table_line_items)
        
        # If no line items found from tables, try text-based extraction
        if not line_items:
            text_line_items = self._extract_text_based_line_items(extracted_data["text_content"])
            line_items.extend(text_line_items)
        
        return line_items
    
    def extract_metadata(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Extract metadata from a PDF document.
        
        Args:
            file_path: Path to the PDF document
            **kwargs: Additional extractor-specific arguments
            
        Returns:
            Dictionary containing document metadata
        """
        logger.info(f"Extracting metadata from PDF: {file_path}")
        
        # Convert to Path object if string
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Initialize metadata dictionary
        metadata = {
            "filename": file_path.name,
            "file_size": file_path.stat().st_size,
            "creation_date": None,
            "modification_date": None,
            "author": None,
            "title": None,
            "subject": None,
            "keywords": None,
            "page_count": 0,
        }
        
        try:
            with pdfplumber.open(file_path) as pdf:
                # Extract PDF metadata
                pdf_meta = pdf.metadata
                if pdf_meta:
                    metadata.update({
                        "author": pdf_meta.get("Author"),
                        "title": pdf_meta.get("Title"),
                        "subject": pdf_meta.get("Subject"),
                        "keywords": pdf_meta.get("Keywords"),
                        "creation_date": pdf_meta.get("CreationDate"),
                        "modification_date": pdf_meta.get("ModDate"),
                    })
                
                # Get page count
                metadata["page_count"] = len(pdf.pages)
                
                # Extract key document dates from first page text
                if pdf.pages:
                    first_page_text = pdf.pages[0].extract_text() or ""
                    metadata.update(self._extract_dates_from_text(first_page_text))
        
        except Exception as e:
            logger.error(f"Error extracting metadata from PDF {file_path}: {str(e)}")
            raise
        
        return metadata

    def _extract_page_data(self, page: Page, page_number: int) -> Dict[str, Any]:
        """
        Extract data from a single PDF page.
        
        Args:
            page: pdfplumber Page object
            page_number: Page number (1-based)
            
        Returns:
            Dictionary containing page data
        """
        # Extract text content
        text = page.extract_text() or ""
        
        # Extract tables from the page
        tables = []
        for table in page.extract_tables():
            # Skip empty tables
            if not table or all(not row for row in table):
                continue
                
            # Clean and normalize table data
            cleaned_table = []
            for row in table:
                cleaned_row = []
                for cell in row:
                    # Normalize cell content
                    if cell is None:
                        cell = ""
                    else:
                        cell = str(cell).strip()
                    cleaned_row.append(cell)
                cleaned_table.append(cleaned_row)
                
            # Determine if this is a header row
            has_header = self._detect_header_row(cleaned_table)
            
            # Try to extract column names if header row exists
            columns = []
            if has_header and len(cleaned_table) > 0:
                columns = cleaned_table[0]
            
            tables.append({
                "page_number": page_number,
                "data": cleaned_table,
                "columns": columns,
                "has_header": has_header,
                "row_count": len(cleaned_table),
                "column_count": len(cleaned_table[0]) if cleaned_table else 0,
            })
            
        # Construct page data dictionary
        page_data = {
            "page_number": page_number,
            "text": text,
            "tables": tables,
            "width": page.width,
            "height": page.height,
        }
        
        return page_data
    
    def _detect_header_row(self, table_data: List[List[str]]) -> bool:
        """
        Detect if the first row of a table is a header row.
        
        Args:
            table_data: Table data as a list of rows
            
        Returns:
            True if the first row appears to be a header, False otherwise
        """
        if not table_data or len(table_data) < 2:
            return False
        
        # Check if the first row has different formatting or content pattern
        first_row = table_data[0]
        
        # If all cells in the first row are capitalized or contain keywords
        header_indicators = ["total", "subtotal", "item", "description", "qty", "quantity", 
                            "amount", "price", "cost", "date", "no.", "number", "unit"]
        
        # Check for header-like formatting or content
        capitals_count = sum(1 for cell in first_row if cell and cell.isupper())
        indicators_count = sum(1 for cell in first_row if cell and 
                             any(indicator in cell.lower() for indicator in header_indicators))
        
        # Return True if first row has more capitalized cells or header indicators
        return (capitals_count > len(first_row) // 2) or (indicators_count > 0)
    
    def _get_table_signature(self, table_data: List[List[str]]) -> str:
        """
        Generate a signature for a table to identify duplicates.
        
        Args:
            table_data: Table data as a list of rows
            
        Returns:
            String signature that represents the table content
        """
        if not table_data:
            return ""
            
        # Create a simple signature based on the first row, last row, and table dimensions
        first_row = "".join(str(cell) for cell in table_data[0])
        last_row = "".join(str(cell) for cell in table_data[-1])
        dimensions = f"{len(table_data)}x{len(table_data[0]) if table_data else 0}"
        
        return f"{dimensions}_{first_row[:50]}_{last_row[:50]}"
    
    def _is_financial_table(self, table_data: List[List[str]]) -> bool:
        """
        Determine if a table contains financial information.
        
        Args:
            table_data: Table data as a list of rows
            
        Returns:
            True if the table appears to be financial, False otherwise
        """
        if not table_data or len(table_data) < 2:
            return False
            
        # Check for financial column headers
        financial_headers = ["amount", "total", "price", "cost", "subtotal", "balance", "sum", 
                            "payment", "claim", "value", "$", "usd", "dollar"]
        
        # Look for currency patterns in the table cells
        currency_pattern = re.compile(r'(\$\s*[\d,.]+)|(\d+\.\d\d)')
        
        # Check first row for financial headers
        header_row = table_data[0]
        has_financial_header = any(
            header and any(fh in header.lower() for fh in financial_headers)
            for header in header_row
        )
        
        # Count cells with currency values
        currency_cells = 0
        for row in table_data[1:]:  # Skip header row
            for cell in row:
                if cell and currency_pattern.search(cell):
                    currency_cells += 1
        
        # Calculate percentage of cells with currency values
        total_cells = sum(len(row) for row in table_data[1:])
        currency_cell_percentage = (currency_cells / total_cells) if total_cells > 0 else 0
        
        # Return True if table has financial headers or significant currency cells
        return has_financial_header or currency_cell_percentage > 0.15
    
    def _extract_financial_line_items(self, table_data: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Extract financial line items from a table.
        
        Args:
            table_data: Table data as a list of rows
            
        Returns:
            List of dictionaries representing financial line items
        """
        line_items = []
        
        # Skip empty tables
        if not table_data or len(table_data) < 2:
            return line_items
        
        # Determine if first row is a header
        has_header = self._detect_header_row(table_data)
        
        # Get column indices for relevant information
        description_idx = None
        amount_idx = None
        quantity_idx = None
        unit_price_idx = None
        
        # Currency pattern for detecting amount columns
        currency_pattern = re.compile(r'(\$\s*[\d,.]+)|(\d+\.\d\d)')
        
        # Try to identify column indices from headers
        if has_header:
            headers = [h.lower() if h else "" for h in table_data[0]]
            
            # Find description column
            for idx, header in enumerate(headers):
                if header and any(kw in header for kw in ["description", "item", "work", "scope"]):
                    description_idx = idx
                    break
            
            # Find amount column
            for idx, header in enumerate(headers):
                if header and any(kw in header for kw in ["amount", "total", "price", "cost", "$"]):
                    amount_idx = idx
                    break
            
            # Find quantity column
            for idx, header in enumerate(headers):
                if header and any(kw in header for kw in ["qty", "quantity", "count"]):
                    quantity_idx = idx
                    break
            
            # Find unit price column
            for idx, header in enumerate(headers):
                if header and any(kw in header for kw in ["unit price", "rate", "unit cost"]):
                    unit_price_idx = idx
                    break
        
        # If headers didn't help, try to identify columns by content patterns
        if amount_idx is None:
            # Find column with most currency values
            currency_counts = [0] * len(table_data[0])
            for row in table_data[1:]:  # Skip header row
                for i, cell in enumerate(row):
                    if cell and currency_pattern.search(cell):
                        currency_counts[i] += 1
            
            if currency_counts:
                # Set the column with most currency values as the amount column
                amount_idx = currency_counts.index(max(currency_counts))
        
        # Process each row (skip header if present)
        start_idx = 1 if has_header else 0
        for row_idx, row in enumerate(table_data[start_idx:], start=start_idx):
            # Skip empty rows
            if not row or all(not cell for cell in row):
                continue
            
            # Skip subtotal/total rows
            if any(row[0].lower().startswith(kw) for kw in ["total", "subtotal", "sum"]):
                continue
            
            # Extract line item data
            line_item = {
                "row_number": row_idx + 1,  # 1-based row number
                "description": "",
                "amount": None,
                "quantity": None,
                "unit_price": None,
                "raw_data": row,
            }
            
            # Extract description
            if description_idx is not None and description_idx < len(row):
                line_item["description"] = row[description_idx]
            else:
                # Try to use first column as description if not identified
                line_item["description"] = row[0] if row else ""
            
            # Extract amount
            if amount_idx is not None and amount_idx < len(row):
                line_item["amount"] = self._extract_currency_value(row[amount_idx])
            
            # Extract quantity
            if quantity_idx is not None and quantity_idx < len(row):
                line_item["quantity"] = self._extract_numeric_value(row[quantity_idx])
            
            # Extract unit price
            if unit_price_idx is not None and unit_price_idx < len(row):
                line_item["unit_price"] = self._extract_currency_value(row[unit_price_idx])
            
            # Add line item if it has meaningful data
            if line_item["description"] or line_item["amount"] is not None:
                line_items.append(line_item)
        
        return line_items
    
    def _extract_text_based_line_items(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract line items from text when table extraction fails.
        
        Args:
            text: Text content from PDF
            
        Returns:
            List of dictionaries representing financial line items
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
    
    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """
        Extract numeric value from text.
        
        Args:
            text: Text containing a numeric value
            
        Returns:
            Extracted float value or None if no valid value found
        """
        if not text:
            return None
        
        # Clean text and remove non-numeric characters (except decimal point and minus)
        cleaned_text = re.sub(r'[^\d.-]', '', text.strip())
        
        # Extract numeric value
        try:
            # Find the first sequence of digits with optional decimal point
            matches = re.search(r'-?\d+\.?\d*', cleaned_text)
            if matches:
                return float(matches.group(0))
            return None
        except (ValueError, TypeError):
            return None
    
    def _extract_dates_from_text(self, text: str) -> Dict[str, str]:
        """
        Extract key dates from document text.
        
        Args:
            text: Text content to search for dates
            
        Returns:
            Dictionary containing extracted dates
        """
        dates = {}
        
        # Common date patterns in construction documents
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
        
        # Search for each date type using patterns
        for date_key, patterns in date_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    dates[date_key] = match.group(1)
                    break
        
        return dates