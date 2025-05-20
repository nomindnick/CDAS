"""
Handwriting recognition module for processing handwritten text in documents.
"""
import logging
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


class HandwritingRecognizer:
    """
    Recognizes handwritten text in images and documents.
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize the handwriting recognizer.
        
        Args:
            tesseract_cmd: Path to the tesseract executable (optional)
        """
        # Set path to tesseract executable if provided
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Verify tesseract installation
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.warning(f"Tesseract not properly configured: {str(e)}")
            logger.warning("Handwriting recognition functionality may not work correctly.")
    
    def recognize_text(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Recognize handwritten text in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing recognition results
        """
        logger.info(f"Recognizing handwritten text in: {image_path}")
        
        # Convert to Path object if string
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        try:
            # Read the image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Detect and process handwritten regions
            processed_image, handwritten_regions = self._detect_handwritten_regions(image)
            
            # Recognize text in each region
            recognition_results = []
            
            for i, region in enumerate(handwritten_regions):
                region_image = self._preprocess_handwriting(region["image"])
                
                # Use Tesseract with handwriting-optimized configuration
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.():;$%-+/ "'
                
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(region_image)
                
                # Perform OCR
                text = pytesseract.image_to_string(pil_image, config=custom_config)
                
                # Get confidence level
                data = pytesseract.image_to_data(pil_image, config=custom_config, output_type=pytesseract.Output.DICT)
                confidences = [conf for conf, word in zip(data["conf"], data["text"]) if word.strip()]
                confidence = sum(confidences) / len(confidences) if confidences else 0
                
                recognition_results.append({
                    "region_id": i + 1,
                    "x": region["x"],
                    "y": region["y"],
                    "width": region["width"],
                    "height": region["height"],
                    "text": text.strip(),
                    "confidence": confidence,
                })
            
            # Combine all recognized text
            full_text = "\n".join(result["text"] for result in recognition_results)
            
            return {
                "text": full_text,
                "regions": recognition_results,
                "region_count": len(recognition_results),
            }
            
        except Exception as e:
            logger.error(f"Error recognizing handwritten text: {str(e)}")
            raise
    
    def _detect_handwritten_regions(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Detect regions containing handwritten text.
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            Tuple of (processed image, list of handwritten regions)
        """
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding to get binary image
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply morphological operations to connect nearby text
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find text regions
        handwritten_regions = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip small regions (likely noise)
            if w < 50 or h < 20:
                continue
            
            # Skip regions that are too large (likely document boundaries)
            if w > image.shape[1] * 0.9 or h > image.shape[0] * 0.9:
                continue
            
            # Check if the region likely contains handwriting
            roi = gray[y:y+h, x:x+w]
            
            if self._is_handwritten(roi):
                # Add margin around the region for better recognition
                margin = 10
                x_margin = max(0, x - margin)
                y_margin = max(0, y - margin)
                w_margin = min(image.shape[1] - x_margin, w + 2 * margin)
                h_margin = min(image.shape[0] - y_margin, h + 2 * margin)
                
                roi_with_margin = gray[y_margin:y_margin+h_margin, x_margin:x_margin+w_margin]
                
                handwritten_regions.append({
                    "x": x_margin,
                    "y": y_margin,
                    "width": w_margin,
                    "height": h_margin,
                    "image": roi_with_margin,
                })
        
        # Create a copy of the original image with regions marked
        marked_image = image.copy()
        for region in handwritten_regions:
            cv2.rectangle(marked_image, (region["x"], region["y"]), 
                         (region["x"] + region["width"], region["y"] + region["height"]), 
                         (0, 255, 0), 2)
        
        return marked_image, handwritten_regions
    
    def _is_handwritten(self, roi: np.ndarray) -> bool:
        """
        Determine if a region contains handwritten text.
        
        Args:
            roi: Region of interest (image patch)
            
        Returns:
            True if the region likely contains handwriting, False otherwise
        """
        # Apply adaptive thresholding
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate the number of non-zero pixels (text pixels)
        text_pixels = cv2.countNonZero(binary)
        
        # Calculate the total number of pixels
        total_pixels = roi.size
        
        # Calculate the text density
        text_density = text_pixels / total_pixels
        
        # Handwritten text typically has lower density than printed text
        if 0.05 <= text_density <= 0.3:
            # Calculate stroke width variation (SWV)
            strokes = self._calculate_stroke_width_variation(binary)
            
            # Handwritten text typically has higher stroke width variation
            return strokes > 0.3
        
        return False
    
    def _calculate_stroke_width_variation(self, binary: np.ndarray) -> float:
        """
        Calculate stroke width variation, which is typically higher for handwritten text.
        
        Args:
            binary: Binary image
            
        Returns:
            Stroke width variation measure
        """
        # Apply distance transform to get stroke widths
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        
        # Get non-zero elements (text pixels)
        stroke_widths = dist[dist > 0]
        
        # Calculate variation
        if len(stroke_widths) > 0:
            mean = np.mean(stroke_widths)
            std = np.std(stroke_widths)
            return std / mean if mean > 0 else 0
        
        return 0
    
    def _preprocess_handwriting(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for better handwriting recognition.
        
        Args:
            image: Image containing handwritten text
            
        Returns:
            Preprocessed image
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply adaptive thresholding to get binary image
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply morphological operations to smooth text
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Perform noise removal
        denoised = cv2.fastNlMeansDenoising(opening, None, 10, 7, 21)
        
        return denoised
    
    def extract_handwritten_amounts(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract handwritten monetary amounts from an image.
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            List of dictionaries containing extracted amounts
        """
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect handwritten regions
        _, handwritten_regions = self._detect_handwritten_regions(gray)
        
        # Extract amount information
        amounts = []
        
        for region in handwritten_regions:
            region_image = self._preprocess_handwriting(region["image"])
            
            # Use Tesseract with digit-optimized configuration
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789,.%$()-+ "'
            text = pytesseract.image_to_string(region_image, config=custom_config)
            
            # Try to extract amount using regular expressions
            import re
            amount_patterns = [
                r'\$\s*[\d,]+\.?\d*',  # $1,000.00
                r'[\d,]+\.?\d*\s*\$',  # 1,000.00$
                r'[\d,]+\.\d{2}',      # 1,000.00
            ]
            
            for pattern in amount_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    for match in matches:
                        # Clean the amount string
                        clean_amount = match.replace('$', '').replace(',', '').strip()
                        try:
                            value = float(clean_amount)
                            amounts.append({
                                "text": match,
                                "value": value,
                                "x": region["x"],
                                "y": region["y"],
                                "width": region["width"],
                                "height": region["height"],
                            })
                        except ValueError:
                            continue
        
        return amounts