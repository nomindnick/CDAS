# Document Processor Specification

## Overview

The Document Processor component is responsible for extracting structured data from various document types relevant to construction disputes. It transforms raw documents (PDFs, Excel spreadsheets, scanned images) into a normalized format that can be stored in the database and analyzed by other system components.

## Key Responsibilities

1. Document ingestion and registration
2. Text and tabular data extraction
3. OCR for scanned documents
4. Handwriting recognition
5. Financial data normalization
6. Metadata preservation
7. Document classification
8. Entity extraction (amounts, dates, references)

## Component Architecture

```
document_processor/
├─ __init__.py
├─ processor.py           # Main processor class
├─ registration.py        # Document registration
├─ extractors/            # Document type extractors
│   ├─ __init__.py
│   ├─ pdf.py             # PDF extraction
│   ├─ excel.py           # Excel extraction
│   ├─ image.py           # Image extraction
│   └─ factory.py         # Extractor factory
├─ ocr.py                 # OCR capabilities
├─ handwriting.py         # Handwriting recognition
├─ normalization.py       # Data normalization
└─ classifiers/           # Document classifiers
    ├─ __init__.py
    ├─ document_type.py   # Document type classification
    └─ content_type.py    # Content classification
```

## Core Classes

### DocumentProcessor

The main entry point for document processing.

```python
class DocumentProcessor:
    """Main document processing orchestrator."""
    
    def __init__(self, config=None, db_session=None):
        """Initialize the document processor.
        
        Args:
            config: Configuration dictionary
            db_session: Database session
        """
        self.config = config or {}
        self.db_session = db_session
        self.extractor_factory = ExtractorFactory(self.config)
    
    def process_document(self, file_path, metadata=None):
        """Process a document file.
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata dictionary
            
        Returns:
            Document object with extracted content
        """
        # 1. Register document
        document = self._register_document(file_path, metadata)
        
        # 2. Select appropriate extractor
        extractor = self.extractor_factory.create_extractor(file_path)
        
        # 3. Extract content
        content = extractor.extract(file_path)
        
        # 4. Normalize data
        normalized_data = self._normalize_data(content)
        
        # 5. Store extracted data
        self._store_extracted_data(document, normalized_data)
        
        return document
    
    def _register_document(self, file_path, metadata):
        """Register a document in the system."""
        # Create document record with hash, type, etc.
        pass
    
    def _normalize_data(self, content):
        """Normalize extracted data."""
        # Apply data normalization rules
        pass
    
    def _store_extracted_data(self, document, normalized_data):
        """Store extracted data in the database."""
        # Save line items and other extracted entities
        pass
```

### ExtractorFactory

Factory for creating appropriate extractors based on file type.

```python
class ExtractorFactory:
    """Factory for creating document extractors."""
    
    def __init__(self, config=None):
        """Initialize the extractor factory.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def create_extractor(self, file_path):
        """Create appropriate extractor for the file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extractor instance
        """
        file_extension = self._get_file_extension(file_path)
        
        if file_extension in ['.pdf']:
            return PDFExtractor(self.config)
        elif file_extension in ['.xlsx', '.xls']:
            return ExcelExtractor(self.config)
        elif file_extension in ['.jpg', '.png', '.tiff']:
            return ImageExtractor(self.config)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _get_file_extension(self, file_path):
        """Get the file extension."""
        return Path(file_path).suffix.lower()
```

### PDFExtractor

Specialized extractor for PDF documents.

```python
class PDFExtractor:
    """Extractor for PDF documents."""
    
    def __init__(self, config=None):
        """Initialize the PDF extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def extract(self, file_path):
        """Extract content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted content dictionary
        """
        # Extract text, tables, and other content
        with pdfplumber.open(file_path) as pdf:
            content = {
                'pages': [],
                'metadata': pdf.metadata,
                'tables': []
            }
            
            # Process each page
            for page_num, page in enumerate(pdf.pages):
                page_content = self._process_page(page, page_num)
                content['pages'].append(page_content)
                
                # Extract tables from the page
                tables = page.extract_tables()
                if tables:
                    for table_num, table in enumerate(tables):
                        table_content = {
                            'page_num': page_num,
                            'table_num': table_num,
                            'data': table
                        }
                        content['tables'].append(table_content)
            
        return content
    
    def _process_page(self, page, page_num):
        """Process a single PDF page."""
        text = page.extract_text()
        
        # Extract financial amounts
        amounts = self._extract_amounts(text)
        
        # Look for handwritten annotations
        annotations = self._detect_handwriting(page)
        
        return {
            'page_num': page_num,
            'text': text,
            'amounts': amounts,
            'annotations': annotations
        }
    
    def _extract_amounts(self, text):
        """Extract financial amounts from text."""
        # Use regex to find dollar amounts
        amount_pattern = r'\$?([0-9,]+\.[0-9]{2})'
        amounts = []
        
        for match in re.finditer(amount_pattern, text):
            # Clean and parse the amount
            amount_str = match.group(1).replace(',', '')
            amount = float(amount_str)
            
            # Get surrounding context
            start_pos = max(0, match.start() - 100)
            end_pos = min(len(text), match.end() + 100)
            context = text[start_pos:end_pos]
            
            amounts.append({
                'amount': amount,
                'context': context,
                'position': match.start()
            })
            
        return amounts
    
    def _detect_handwriting(self, page):
        """Detect handwritten annotations on the page."""
        # Detect handwritten annotations
        # This would use image processing techniques
        # or AI models designed for handwriting detection
        return []
```

### ExcelExtractor

Specialized extractor for Excel spreadsheets.

```python
class ExcelExtractor:
    """Extractor for Excel documents."""
    
    def __init__(self, config=None):
        """Initialize the Excel extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def extract(self, file_path):
        """Extract content from an Excel file.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Extracted content dictionary
        """
        content = {
            'sheets': [],
            'line_items': []
        }
        
        # Read Excel file
        excel_file = pd.ExcelFile(file_path)
        
        # Process each sheet
        for sheet_name in excel_file.sheet_names:
            # Read the sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Clean and normalize the dataframe
            df_cleaned = self._clean_dataframe(df)
            
            # Extract line items
            line_items = self._extract_line_items(df_cleaned, sheet_name)
            
            # Store sheet content
            sheet_content = {
                'sheet_name': sheet_name,
                'data': df_cleaned.to_dict(),
                'line_items': line_items
            }
            
            content['sheets'].append(sheet_content)
            content['line_items'].extend(line_items)
            
        return content
    
    def _clean_dataframe(self, df):
        """Clean and normalize a dataframe."""
        # Handle merged cells by forward-filling
        df = df.fillna(method='ffill')
        
        # Drop completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Handle multi-row headers (if detected)
        if self._has_multi_row_header(df):
            df = self._process_multi_row_header(df)
            
        return df
    
    def _has_multi_row_header(self, df):
        """Detect if dataframe has multi-row header."""
        # Heuristic: check if first few rows contain mostly strings
        # and later rows contain mostly numbers
        pass
    
    def _process_multi_row_header(self, df):
        """Process dataframe with multi-row header."""
        # Combine header rows into a single header
        pass
    
    def _extract_line_items(self, df, sheet_name):
        """Extract line items from a dataframe."""
        line_items = []
        
        # Identify likely description and amount columns
        desc_col = self._find_description_column(df)
        amount_cols = self._find_amount_columns(df)
        
        # Extract line items
        for idx, row in df.iterrows():
            # Skip header-like rows
            if self._is_header_row(row):
                continue
                
            description = str(row[desc_col]) if pd.notna(row[desc_col]) else ""
            
            # Extract amounts from amount columns
            for col in amount_cols:
                if pd.notna(row[col]) and isinstance(row[col], (int, float)):
                    amount = float(row[col])
                    
                    # Create line item
                    line_item = {
                        'description': description,
                        'amount': amount,
                        'sheet_name': sheet_name,
                        'row_index': idx,
                        'column_name': col
                    }
                    
                    line_items.append(line_item)
        
        return line_items
    
    def _find_description_column(self, df):
        """Find the most likely description column."""
        # Heuristic: first column or column with mostly string values
        pass
    
    def _find_amount_columns(self, df):
        """Find columns likely to contain dollar amounts."""
        # Heuristic: columns with mostly numeric values
        pass
    
    def _is_header_row(self, row):
        """Check if a row appears to be a header row."""
        # Heuristic: row with mostly string values and/or
        # row with terms like "total", "subtotal", etc.
        pass
```

## OCR Integration

For scanned documents, the system will use OCR to extract text.

```python
def process_image_ocr(image_path):
    """Process an image file using OCR.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text
    """
    # Use pytesseract to perform OCR
    import pytesseract
    from PIL import Image
    
    # Preprocess image
    image = Image.open(image_path)
    
    # Perform OCR
    text = pytesseract.image_to_string(image)
    
    return text
```

## Handwriting Recognition

For documents with handwritten annotations, the system will use specialized models for handwriting recognition.

```python
def detect_handwritten_regions(image):
    """Detect regions containing handwriting.
    
    Args:
        image: Image or page object
        
    Returns:
        List of regions with handwriting
    """
    # Use computer vision techniques to identify
    # regions that likely contain handwriting
    pass

def extract_handwriting(region):
    """Extract text from handwritten region.
    
    Args:
        region: Image region with handwriting
        
    Returns:
        Extracted text
    """
    # Use specialized handwriting recognition model
    # (e.g., Google Cloud Vision API, Azure Computer Vision)
    pass
```

## Document Classification

The system will automatically classify documents based on their content.

```python
def classify_document(content):
    """Classify document type based on content.
    
    Args:
        content: Extracted document content
        
    Returns:
        Document classification
    """
    # Use rule-based or ML-based classification
    # to determine document type (payment application,
    # change order, etc.)
    pass
```

## Financial Data Normalization

The system will normalize financial data from various formats.

```python
def normalize_financial_data(data):
    """Normalize financial data.
    
    Args:
        data: Raw financial data
        
    Returns:
        Normalized financial data
    """
    # Convert amounts to standard format
    # Handle different formats (e.g., $1,234.56, 1234.56, etc.)
    pass
```

## Implementation Guidelines

1. **Modularity**: Each extractor should be self-contained and follow a common interface
2. **Error Handling**: Robust error handling for various document formats and quality issues
3. **Performance**: Optimize for performance with large documents
4. **Configuration**: Support configuration options for different extraction approaches
5. **Extensibility**: Design for easy addition of new document types and extraction methods

## Dependencies

- pdfplumber: PDF text and table extraction
- pandas: Excel processing and data manipulation
- pytesseract: OCR for scanned documents
- Pillow: Image processing
- numpy: Numerical processing
- regex: Regular expression support
- langchain: Optional LLM integration for complex extraction

## Testing Strategy

1. Unit tests for each extractor
2. Integration tests with sample documents
3. Benchmark tests for performance
4. Validation tests for accuracy of extraction

## Security Considerations

1. Validate all input files before processing
2. Scan for malicious content
3. Ensure secure handling of sensitive information
4. Implement proper permissions for document access
