"""
OCR (Optical Character Recognition) module for processing scanned documents.
"""
import os
import logging
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path
import tempfile

import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    Processes scanned documents and images using OCR.
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None, dpi: int = 300):
        """
        Initialize the OCR processor.
        
        Args:
            tesseract_cmd: Path to the tesseract executable (optional)
            dpi: DPI to use when converting PDFs to images (higher means better quality but slower)
        """
        self.dpi = dpi
        
        # Set path to tesseract executable if provided
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Verify tesseract installation
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.warning(f"Tesseract not properly configured: {str(e)}")
            logger.warning("OCR functionality may not work correctly.")
    
    def process_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a single image with OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing OCR results
        """
        logger.info(f"Processing image with OCR: {image_path}")
        
        # Convert to Path object if string
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        try:
            # Read the image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Preprocess the image for better OCR results
            preprocessed = self._preprocess_image(image)
            
            # Perform OCR
            ocr_result = self._perform_ocr(preprocessed)
            
            # Return results
            return {
                "text": ocr_result["text"],
                "words": ocr_result["words"],
                "confidence": ocr_result["confidence"],
            }
            
        except Exception as e:
            logger.error(f"Error processing image with OCR: {str(e)}")
            raise
    
    def process_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a PDF document with OCR.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing OCR results for each page
        """
        logger.info(f"Processing PDF with OCR: {pdf_path}")
        
        # Convert to Path object if string
        if isinstance(pdf_path, str):
            pdf_path = Path(pdf_path)
        
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=self.dpi)
            
            # Process each page
            pages = []
            full_text = ""
            
            for i, image in enumerate(images):
                # Convert PIL Image to OpenCV format
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Preprocess the image
                preprocessed = self._preprocess_image(opencv_image)
                
                # Perform OCR
                ocr_result = self._perform_ocr(preprocessed)
                
                # Add page result
                page_result = {
                    "page_number": i + 1,
                    "text": ocr_result["text"],
                    "words": ocr_result["words"],
                    "confidence": ocr_result["confidence"],
                }
                
                pages.append(page_result)
                full_text += ocr_result["text"] + "\n\n"
            
            # Return results
            return {
                "text": full_text,
                "pages": pages,
                "page_count": len(pages),
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF with OCR: {str(e)}")
            raise
    
    def detect_form_fields(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect form fields (checkboxes, text fields) in an image.
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            List of dictionaries, each representing a detected form field
        """
        # Create a copy of the image for drawing
        form_fields = []
        
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find form fields
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip small contours (noise)
            if w < 20 or h < 10:
                continue
            
            # Calculate aspect ratio
            aspect_ratio = float(w) / h
            
            # Classify the contour based on its characteristics
            field_type = self._classify_form_field(gray[y:y+h, x:x+w], aspect_ratio, w, h)
            
            if field_type:
                # Extract text in the vicinity if it's a form field
                field_text = ""
                if field_type in ["text_field", "checkbox"]:
                    # Look for text labels to the left or above the field
                    left_text = self._extract_text_region(gray, max(0, x-200), y, min(200, x), h)
                    top_text = self._extract_text_region(gray, x, max(0, y-50), w, min(50, y))
                    
                    field_text = left_text or top_text
                
                form_fields.append({
                    "type": field_type,
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "label": field_text.strip(),
                })
        
        return form_fields
    
    def extract_tables_from_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract tables from an image.
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            List of dictionaries, each representing a table
        """
        # Initialize results
        tables = []
        
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Apply morphological operations to find horizontal and vertical lines
        kernel_length = np.array(gray).shape[1] // 80
        
        # Vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        vertical_lines = cv2.erode(binary, vertical_kernel, iterations=3)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=3)
        
        # Horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        horizontal_lines = cv2.erode(binary, horizontal_kernel, iterations=3)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=3)
        
        # Combine horizontal and vertical lines
        table_structure = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        
        # Find contours of the combined lines
        contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour as a potential table
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small contours (too small to be tables)
            if w < 100 or h < 100:
                continue
            
            # Extract the table region
            table_region = gray[y:y+h, x:x+w]
            
            # Use Tesseract to extract text and tables from this region
            try:
                # Use Tesseract's built-in table extraction capability
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.():;$%-+/ "'
                table_data = pytesseract.image_to_data(table_region, config=custom_config, output_type=pytesseract.Output.DICT)
                
                # Convert to a simpler structure
                table = self._convert_tesseract_data_to_table(table_data)
                
                if table:
                    table["x"] = x
                    table["y"] = y
                    table["width"] = w
                    table["height"] = h
                    tables.append(table)
            except Exception as e:
                logger.error(f"Error extracting table data: {str(e)}")
        
        return tables
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for better OCR results.
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #                                cv2.THRESH_BINARY, 11, 2)
        
        # Apply dilation to make text more visible
        # kernel = np.ones((1, 1), np.uint8)
        # dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        return denoised
    
    def _perform_ocr(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform OCR on a preprocessed image.
        
        Args:
            image: Preprocessed OpenCV image (numpy array)
            
        Returns:
            Dictionary containing OCR results
        """
        # Convert OpenCV image to PIL Image
        pil_image = Image.fromarray(image)
        
        # Perform OCR
        text = pytesseract.image_to_string(pil_image)
        
        # Get detailed OCR data (word-level)
        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        
        # Extract words with confidence
        words = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            # Skip empty words
            if not data['text'][i].strip():
                continue
                
            words.append({
                'text': data['text'][i],
                'confidence': data['conf'][i],
                'x': data['left'][i],
                'y': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i],
                'block_num': data['block_num'][i],
                'line_num': data['line_num'][i],
            })
        
        # Calculate overall confidence
        confidence = sum(word['confidence'] for word in words) / len(words) if words else 0
        
        return {
            "text": text,
            "words": words,
            "confidence": confidence,
        }
    
    def _classify_form_field(self, roi: np.ndarray, aspect_ratio: float, width: int, height: int) -> Optional[str]:
        """
        Classify a region of interest as a form field.
        
        Args:
            roi: Region of interest (image patch)
            aspect_ratio: Width/height ratio
            width: Width of the ROI
            height: Height of the ROI
            
        Returns:
            Field type as string, or None if not a form field
        """
        # Checkbox detection (small square)
        if 0.8 <= aspect_ratio <= 1.2 and 10 <= width <= 50 and 10 <= height <= 50:
            return "checkbox"
        
        # Text field detection (long rectangle)
        if aspect_ratio > 3 and width > 100:
            return "text_field"
        
        # Signature field detection (medium to large rectangle)
        if 2 <= aspect_ratio <= 6 and width > 150 and height > 30:
            return "signature_field"
        
        return None
    
    def _extract_text_region(self, image: np.ndarray, x: int, y: int, width: int, height: int) -> str:
        """
        Extract text from a specific region of the image.
        
        Args:
            image: Image as numpy array
            x, y, width, height: Region coordinates
            
        Returns:
            Extracted text
        """
        # Extract the region
        region = image[y:y+height, x:x+width]
        
        # Convert to PIL Image
        pil_region = Image.fromarray(region)
        
        # Perform OCR
        try:
            text = pytesseract.image_to_string(pil_region)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from region: {str(e)}")
            return ""
    
    def _convert_tesseract_data_to_table(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Tesseract data output to a structured table.
        
        Args:
            data: Tesseract output dictionary
            
        Returns:
            Dictionary representing the table structure
        """
        # Initialize table structure
        table = {
            "rows": [],
            "row_count": 0,
            "column_count": 0,
        }
        
        # Get unique block numbers (each block might represent a row or cell)
        block_nums = sorted(set(data['block_num']))
        
        # Skip the first block (usually 0, which is for non-block content)
        if block_nums and block_nums[0] == 0:
            block_nums = block_nums[1:]
        
        # Parse blocks into rows
        current_row = []
        current_row_num = None
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text:
                continue
                
            block_num = data['block_num'][i]
            line_num = data['line_num'][i]
            
            # If this is the start of a new row
            if current_row_num is None or line_num != current_row_num:
                if current_row:
                    table["rows"].append(current_row)
                
                current_row = [text]
                current_row_num = line_num
            else:
                # Continue the current row
                current_row.append(text)
        
        # Add the last row if not empty
        if current_row:
            table["rows"].append(current_row)
        
        # Update metadata
        table["row_count"] = len(table["rows"])
        table["column_count"] = max(len(row) for row in table["rows"]) if table["rows"] else 0
        
        return table