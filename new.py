import streamlit as st
import cv2
import numpy as np
import pytesseract
import re
import json
from collections import defaultdict
import os

# Try to import NLP libraries with proper error handling
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    # Download NLTK resources with error handling
    def download_nltk_resources():
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt')
            except Exception as e:
                st.warning(f"Failed to download NLTK punkt: {e}")
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            try:
                nltk.download('stopwords')
            except Exception as e:
                st.warning(f"Failed to download NLTK stopwords: {e}")
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    st.warning("NLTK not available. Some text processing features will be limited.")

# Try to import SpellChecker with error handling
try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    st.warning("SpellChecker not available. Spell checking will be disabled.")

# Function for advanced image preprocessing
def preprocess_image(input_image, options):
    """
    Enhanced preprocessing of the image with multiple CV techniques for optimal OCR.
    """
    # Create a copy of the original image
    processed = input_image.copy()
    
    # Convert to grayscale if enabled
    if options['use_grayscale']:
        if len(processed.shape) == 3:  # Only convert if it's a color image
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # Apply noise reduction techniques
    if options['use_denoise']:
        if len(processed.shape) == 3:  # Color image
            try:
                processed = cv2.fastNlMeansDenoisingColored(processed, None, options['denoise_strength'], 
                                                           options['denoise_strength'], 7, 21)
            except Exception as e:
                st.warning(f"Denoising error: {e}. Using bilateral filter instead.")
                processed = cv2.bilateralFilter(processed, 9, 75, 75)
        else:  # Grayscale image
            try:
                processed = cv2.fastNlMeansDenoising(processed, None, options['denoise_strength'], 7, 21)
            except Exception as e:
                st.warning(f"Denoising error: {e}. Using median blur instead.")
                processed = cv2.medianBlur(processed, 5)
    
    # Apply Gaussian blur to reduce noise if enabled
    if options['use_blur']:
        processed = cv2.GaussianBlur(processed, (options['blur_kernel_size'], options['blur_kernel_size']), 0)
    
    # Apply unsharp masking for sharpening
    if options['use_sharpen']:
        if len(processed.shape) == 3 or len(processed.shape) == 2:  # Works for both color and grayscale
            blurred = cv2.GaussianBlur(processed, (0, 0), options['sharpen_sigma'])
            processed = cv2.addWeighted(processed, 1 + options['sharpen_intensity'], 
                                       blurred, -options['sharpen_intensity'], 0)
    
    # Adjust brightness and contrast
    processed = cv2.convertScaleAbs(processed, alpha=options['contrast'], beta=options['brightness'])
    
    # Apply dilation and erosion to enhance text
    if options['use_morphology']:
        if len(processed.shape) == 3:  # Ensure grayscale for morphological operations
            gray_for_morph = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray_for_morph = processed.copy()
        
        kernel = np.ones((options['morph_kernel_size'], options['morph_kernel_size']), np.uint8)
        
        if options['morph_operation'] == 'dilate':
            morph_result = cv2.dilate(gray_for_morph, kernel, iterations=options['morph_iterations'])
        elif options['morph_operation'] == 'erode':
            morph_result = cv2.erode(gray_for_morph, kernel, iterations=options['morph_iterations'])
        elif options['morph_operation'] == 'open':
            morph_result = cv2.morphologyEx(gray_for_morph, cv2.MORPH_OPEN, kernel)
        elif options['morph_operation'] == 'close':
            morph_result = cv2.morphologyEx(gray_for_morph, cv2.MORPH_CLOSE, kernel)
        
        # If original was color and we performed grayscale morphology, we need to decide
        # whether to keep color or convert everything to grayscale
        if len(processed.shape) == 3:
            # Option 1: Replace intensity in original color image
            # This keeps color while applying morphological changes to structure
            hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
            hsv[:,:,2] = morph_result  # Replace value channel with morphology result
            processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            processed = morph_result
    
    # Ensure image is grayscale before thresholding
    if options['threshold_method'] != 'none':
        if len(processed.shape) == 3:
            gray_for_threshold = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray_for_threshold = processed.copy()
        
        # Apply thresholding techniques
        if options['threshold_method'] == 'adaptive':
            threshold_result = cv2.adaptiveThreshold(gray_for_threshold, 255, 
                                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY, 
                                                   options['adaptive_block_size'], 
                                                   options['adaptive_c'])
        elif options['threshold_method'] == 'otsu':
            _, threshold_result = cv2.threshold(gray_for_threshold, 0, 255, 
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif options['threshold_method'] == 'binary':
            _, threshold_result = cv2.threshold(gray_for_threshold, 
                                              options['binary_threshold'], 255, 
                                              cv2.THRESH_BINARY)
        
        processed = threshold_result
    
    # Apply edge enhancement if enabled
    if options['use_edge_enhancement']:
        # Ensure we're working with grayscale for edge detection
        if len(processed.shape) == 3:
            gray_for_edge = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray_for_edge = processed.copy()
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray_for_edge, options['canny_threshold1'], options['canny_threshold2'])
        
        # Use edges as a mask for the original image
        if len(input_image.shape) == 3:  # If original was color
            processed = cv2.bitwise_and(input_image, input_image, mask=edges)
        else:
            processed = cv2.bitwise_and(gray_for_edge, gray_for_edge, mask=edges)
    
    return processed

# Function to detect and correct skew
def deskew_image(image):
    """
    Detects and corrects skew in the image for better OCR results.
    """
    try:
        # Ensure image is grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find all contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the rotated rectangles
        angles = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                if -45 <= angle <= 45:  # Filter extreme angles
                    angles.append(angle)
        
        # Calculate the average angle if angles were found
        if angles:
            avg_angle = np.mean(angles)
            # Adjust angle reference
            if avg_angle < -45:
                avg_angle += 90
            
            # Rotate the image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                    borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image  # Return original if no skew detected
    except Exception as e:
        st.warning(f"Deskew error: {e}. Using original image.")
        return image

# Function to extract regions of interest (ROI)
def extract_roi(image):
    """
    Extract potential menu item regions from the image.
    """
    try:
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply dilation to connect nearby text
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours from top to bottom
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        
        # Sort by Y-coordinate (top to bottom)
        sorted_boxes = sorted(zip(contours, bounding_boxes), key=lambda b: b[1][1])
        
        if sorted_boxes:
            contours, bounding_boxes = zip(*sorted_boxes)
        else:
            return [gray]  # Return the whole image as one ROI if no contours found
        
        rois = []
        for (i, (x, y, w, h)) in enumerate(bounding_boxes):
            # Filter out very small contours
            if w > 100 and h > 20:
                roi = gray[y:y+h, x:x+w]
                rois.append(roi)
        
        return rois if rois else [gray]  # Return whole image if no suitable ROIs found
    except Exception as e:
        st.warning(f"ROI extraction error: {e}. Using whole image.")
        if len(image.shape) == 3:
            return [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)]
        return [image]

# Advanced OCR function with custom configuration
def extract_text_from_image(image, config=None):
    """
    Enhanced OCR function with custom Tesseract configuration.
    """
    try:
        if config is None:
            config = '--psm 6 --oem 3'  # Default configuration
        
        extracted_text = pytesseract.image_to_string(image, config=config)
        return extracted_text
    except Exception as e:
        st.error(f"OCR error: {e}. Make sure Tesseract is installed and properly configured.")
        return "OCR ERROR: Failed to extract text. Check Tesseract installation."

# Function to clean and correct OCR text
def clean_ocr_text(text, spell_check=True):
    """
    Cleans and corrects OCR text using NLP techniques.
    """
    # Basic text cleaning without NLP libraries
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove non-alphanumeric characters except for $
    text = re.sub(r'[^\w\s$.]', '', text)
    
    # If NLTK is available and spell checking is requested
    if NLTK_AVAILABLE and SPELLCHECKER_AVAILABLE and spell_check:
        try:
            # Download resources if needed
            download_nltk_resources()
            
            spell = SpellChecker()
            words = word_tokenize(text)
            corrected_words = []
            
            # Common menu words to ignore in spell checking
            menu_words = ["chicken", "burger", "salad", "pizza", "pasta", "steak", 
                         "soup", "sandwich", "appetizer", "beverage", "dessert", 
                         "breakfast", "lunch", "dinner", "special", "combo",
                         "palak", "paneer", "vindaloo", "jerk", "butter", "pork", "spicy"]
            
            for word in words:
                # Skip numbers, dollar signs, and menu-specific words
                if word.isdigit() or '$' in word or word.lower() in menu_words:
                    corrected_words.append(word)
                else:
                    # Check if the word is misspelled
                    misspelled = spell.unknown([word])
                    if misspelled:
                        # Get the most likely correction
                        corrected = spell.correction(word)
                        corrected_words.append(corrected if corrected else word)
                    else:
                        corrected_words.append(word)
            
            text = ' '.join(corrected_words)
        except Exception as e:
            st.warning(f"Spell checking error: {e}. Using basic text cleaning.")
    
    return text

# Function to extract text with bounding boxes using pytesseract
def extract_text_with_boxes(image):
    """
    Extract text with bounding box information using pytesseract.
    Returns both the text and the bounding box coordinates.
    """
    try:
        # Convert image to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Get text and bounding box data
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        
        # Combine the data into a more usable format
        boxes = []
        for i in range(len(data['text'])):
            if data['text'][i].strip():  # Only include non-empty text
                box = {
                    'text': data['text'][i],
                    'conf': data['conf'][i],
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i], 
                    'h': data['height'][i],
                    'line_num': data['line_num'][i],
                    'block_num': data['block_num'][i]
                }
                boxes.append(box)
                
        return boxes
    except Exception as e:
        st.error(f"Error extracting text with boxes: {e}")
        return []

# Function to parse menu based on spatial layout
def parse_menu_with_layout(image):
    """
    Parse the menu using spatial layout analysis with bounding boxes.
    """
    try:
        # Extract text with bounding box information
        boxes = extract_text_with_boxes(image)
        
        if not boxes:
            return json.dumps({"error": "No text extracted from image"}, indent=4)
        
        # Sort boxes by vertical position (y-coordinate) to analyze in reading order
        boxes.sort(key=lambda box: box['y'])
        
        # Identify hotel name (usually at the top of the menu)
        hotel_name = None
        for box in boxes[:5]:  # Check first few text elements
            if "HOTEL" in box['text'].upper():
                hotel_name = box['text']
                break
        
        # Identify menu categories and items
        menu = {}
        current_category = None
        menu_items = []
        price_pattern = re.compile(r'^\$?\s*(\d+)$')
        
        # Group boxes by approximate line (y-coordinate)
        lines = {}
        for box in boxes:
            # Use a tolerance to group boxes that are roughly on the same line
            line_key = box['y'] // 10 * 10  # Round to nearest 10px
            if line_key not in lines:
                lines[line_key] = []
            lines[line_key].append(box)
        
        # Sort lines by y-coordinate
        sorted_lines = sorted(lines.items())
        
        # Process lines to identify categories, items, descriptions, and prices
        item_info = {}
        current_item = None
        
        for line_y, line_boxes in sorted_lines:
            # Sort boxes in this line by x-coordinate
            line_boxes.sort(key=lambda box: box['x'])
            
            line_text = " ".join([box['text'] for box in line_boxes])
            
            # Check if this is a category header
            if all(c.isupper() or c.isspace() for c in line_text) and line_text.strip() and not re.search(r'\$\d+', line_text):
                category_candidates = ["MAIN FOOD", "FOOD", "DRINK", "DESSERT", "ADDITIONAL"]
                for candidate in category_candidates:
                    if candidate in line_text:
                        current_category = candidate
                        if current_category not in menu:
                            menu[current_category] = []
                        break
                continue
            
            # Check if this line has a price (usually at the right side)
            has_price = False
            price_value = None
            item_text = None
            
            # Price is usually the rightmost element on the line
            for box in reversed(line_boxes):
                price_match = price_pattern.search(box['text'])
                if price_match or '$' in box['text']:
                    has_price = True
                    # Extract price value
                    if price_match:
                        price_value = int(price_match.group(1))
                    else:
                        # Try to extract the price from text containing $
                        price_text = re.search(r'\$\s*(\d+)', box['text'])
                        if price_text:
                            price_value = int(price_text.group(1))
                    break
            
            # If we found a price, this is likely an item line
            if has_price and price_value is not None:
                # Get item name from the same line (exclude the price part)
                item_boxes = [b for b in line_boxes if b['text'] != box['text']]
                if item_boxes:
                    item_text = " ".join([b['text'] for b in item_boxes]).strip()
                    
                if item_text and current_category:
                    # Store the item with its price
                    current_item = {
                        "item": item_text,
                        "price": price_value,
                        "description": ""
                    }
                    if current_category in menu:
                        menu[current_category].append(current_item)
            
            # If this line doesn't have a price and we have a current item,
            # it might be a description for the previous item
            elif current_item is not None and current_category in menu:
                # Make sure this line isn't a category header
                if not all(c.isupper() or c.isspace() for c in line_text):
                    # Add to the description of the current item
                    description_text = " ".join([box['text'] for box in line_boxes]).strip()
                    if description_text:
                        current_item["description"] += description_text + " "
        
        # Clean up descriptions (remove trailing space)
        for category in menu:
            for item in menu[category]:
                item["description"] = item["description"].strip()
        
        # Add hotel name if found
        result = {"menu": menu}
        if hotel_name:
            result["hotel_name"] = hotel_name
        
        return json.dumps(result, indent=4)
    except Exception as e:
        st.error(f"Menu layout parsing error: {e}")
        return json.dumps({"error": str(e)}, indent=4)

# Updated main function to incorporate layout analysis
def process_menu_image(image):
    """
    Process menu image using layout analysis approach.
    """
    # First try preprocessing to improve text extraction
    # Get preprocessing options from session state
    options = {
        'use_grayscale': st.session_state.get('use_grayscale', True),
        'use_denoise': st.session_state.get('use_denoise', True),
        'denoise_strength': st.session_state.get('denoise_strength', 10),
        'use_blur': st.session_state.get('use_blur', False),
        'blur_kernel_size': st.session_state.get('blur_kernel_size', 3),
        'use_sharpen': st.session_state.get('use_sharpen', True),
        'sharpen_intensity': st.session_state.get('sharpen_intensity', 1.5),
        'sharpen_sigma': st.session_state.get('sharpen_sigma', 1.0),
        'brightness': st.session_state.get('brightness', 10),
        'contrast': st.session_state.get('contrast', 1.2),
        'use_morphology': st.session_state.get('use_morphology', False),
        'morph_operation': st.session_state.get('morph_operation', 'close'),
        'morph_kernel_size': st.session_state.get('morph_kernel_size', 3),
        'morph_iterations': st.session_state.get('morph_iterations', 1),
        'threshold_method': st.session_state.get('threshold_method', 'adaptive'),
        'binary_threshold': st.session_state.get('binary_threshold', 127),
        'adaptive_block_size': st.session_state.get('adaptive_block_size', 11),
        'adaptive_c': st.session_state.get('adaptive_c', 2),
        'use_edge_enhancement': st.session_state.get('use_edge_enhancement', False),
        'canny_threshold1': st.session_state.get('canny_threshold1', 100),
        'canny_threshold2': st.session_state.get('canny_threshold2', 200),
    }
    
    # Apply deskew if enabled
    if st.session_state.get('auto_deskew', True):
        try:
            image = deskew_image(image)
        except Exception as e:
            st.warning(f"Deskew error: {e}")
    
    # Preprocess the image
    try:
        preprocessed_image = preprocess_image(image, options)
        
        # Parse menu using layout analysis
        result = parse_menu_with_layout(preprocessed_image)
        
        return result, preprocessed_image
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return json.dumps({"error": str(e)}, indent=4), image

# Streamlit UI
st.title("Advanced Hotel Bill OCR Processor")
st.sidebar.header("Processing Options")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["OCR Processor", "Advanced Settings", "About"])

with tab1:
    # Check for Tesseract installation
    try:
        pytesseract_version = pytesseract.get_tesseract_version()
        st.success(f"Tesseract OCR version {pytesseract_version} detected")
    except Exception as e:
        st.error(f"Tesseract OCR not properly configured: {e}")
        st.markdown("""
        ### Tesseract OCR Installation:
        1. Download and install Tesseract from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
        2. Add Tesseract to your PATH or set the path explicitly:
        \`\`\`python
        pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        \`\`\`
        """)
    
    # Upload an image
    uploaded_image = st.file_uploader("Upload a hotel bill image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        try:
            # Convert uploaded image to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display original image
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Apply deskew if enabled
            if st.sidebar.checkbox("Auto-Deskew Image", value=True, key="sidebar_deskew_checkbox"):
                try:
                    deskewed_image = deskew_image(image)
                    st.subheader("Deskewed Image")
                    display_image = cv2.cvtColor(deskewed_image, cv2.COLOR_BGR2RGB) if len(deskewed_image.shape) == 3 else deskewed_image
                    st.image(display_image, use_column_width=True)
                    image = deskewed_image  # Use the deskewed image for further processing
                except Exception as e:
                    st.warning(f"Deskew error: {e}")
            
            # Process button
            if st.button("Process Image", key="process_image_button"):
                with st.spinner("Processing image..."):
                    # Initialize session state for options if not already set
                    if 'use_grayscale' not in st.session_state:
                        st.session_state['use_grayscale'] = True
                    if 'use_denoise' not in st.session_state:
                        st.session_state['use_denoise'] = True
                    if 'denoise_strength' not in st.session_state:
                        st.session_state['denoise_strength'] = 10
                    
                    # Get processing options with defaults
                    options = {
                        'use_grayscale': st.session_state.get('use_grayscale', True),
                        'use_denoise': st.session_state.get('use_denoise', True),
                        'denoise_strength': st.session_state.get('denoise_strength', 10),
                        'use_blur': st.session_state.get('use_blur', False),
                        'blur_kernel_size': st.session_state.get('blur_kernel_size', 3),
                        'use_sharpen': st.session_state.get('use_sharpen', True),
                        'sharpen_intensity': st.session_state.get('sharpen_intensity', 1.5),
                        'sharpen_sigma': st.session_state.get('sharpen_sigma', 1.0),
                        'brightness': st.session_state.get('brightness', 10),
                        'contrast': st.session_state.get('contrast', 1.2),
                        'use_morphology': st.session_state.get('use_morphology', False),
                        'morph_operation': st.session_state.get('morph_operation', 'close'),
                        'morph_kernel_size': st.session_state.get('morph_kernel_size', 3),
                        'morph_iterations':  st.session_state.get('morph_kernel_size', 3),
                        'morph_iterations': st.session_state.get('morph_iterations', 1),
                        'threshold_method': st.session_state.get('threshold_method', 'adaptive'),
                        'binary_threshold': st.session_state.get('binary_threshold', 127),
                        'adaptive_block_size': st.session_state.get('adaptive_block_size', 11),
                        'adaptive_c': st.session_state.get('adaptive_c', 2),
                        'use_edge_enhancement': st.session_state.get('use_edge_enhancement', False),
                        'canny_threshold1': st.session_state.get('canny_threshold1', 100),
                        'canny_threshold2': st.session_state.get('canny_threshold2', 200),
                    }
                    
                    try:
                        # Preprocess image
                        preprocessed_image = preprocess_image(image, options)
                        
                        # Display preprocessed image
                        st.subheader("Preprocessed Image")
                        st.image(preprocessed_image, use_column_width=True)
                        
                        # Set default OCR config if not already set
                        if 'tesseract_config' not in st.session_state:
                            st.session_state['tesseract_config'] = '--psm 6 --oem 3'
                        
                        # Option to process whole image or extract ROIs
                        if st.session_state.get('use_roi', False):
                            rois = extract_roi(preprocessed_image)
                            st.subheader(f"Detected {len(rois)} Regions of Interest")
                            
                            # Display ROIs and extract text from each
                            all_text = []
                            for i, roi in enumerate(rois):
                                if st.session_state.get('show_rois', False) and i < 10:  # Limit to 10 ROIs for display
                                    st.image(roi, caption=f"ROI {i+1}", width=300)
                                
                                # Extract text from ROI with custom config
                                roi_text = extract_text_from_image(roi, st.session_state.get('tesseract_config'))
                                all_text.append(roi_text)
                            
                            # Combine texts
                            extracted_text = "\n".join(all_text)
                        else:
                            # Extract text from whole image
                            extracted_text = extract_text_from_image(
                                preprocessed_image, 
                                st.session_state.get('tesseract_config')
                            )
                        
                        # Clean OCR text if enabled
                        cleaned_text = clean_ocr_text(
                            extracted_text, 
                            spell_check=st.session_state.get('use_spell_check', True) and SPELLCHECKER_AVAILABLE and NLTK_AVAILABLE
                        )
                        
                        # Display extracted text
                        st.subheader("Extracted Text")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text_area("Raw OCR Output", extracted_text, height=200)
                        with col2:
                            st.text_area("Cleaned Text", cleaned_text, height=200)
                        
                        # Parse structured hotel menu
                        structured_menu = parse_menu_with_layout(preprocessed_image)
                        
                        # Display structured JSON output
                        st.subheader("Structured Menu JSON")
                        st.json(structured_menu)
                        
                        # Option to download JSON
                        st.download_button(
                            label="Download JSON",
                            data=structured_menu,
                            file_name="hotel_menu.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"Processing error: {e}")
                        st.info("Try adjusting the preprocessing parameters or check your Tesseract OCR installation.")
        except Exception as e:
            st.error(f"Error loading image: {e}")

with tab2:
    st.header("Image Preprocessing Settings")
    
    st.subheader("OCR Settings")
    st.session_state['auto_deskew'] = st.checkbox("Auto-Deskew Image", value=True, key="auto_deskew_checkbox")
    
    # Basic settings
    st.subheader("Basic Settings")
    st.session_state['use_grayscale'] = st.checkbox("Use Grayscale", value=True)
    st.session_state['brightness'] = st.slider("Brightness", -100, 100, 10)
    st.session_state['contrast'] = st.slider("Contrast", 0.1, 3.0, 1.2)
    
    # Noise reduction
    st.subheader("Noise Reduction")
    st.session_state['use_denoise'] = st.checkbox("Apply Denoising", value=True)
    st.session_state['denoise_strength'] = st.slider("Denoise Strength", 1, 20, 10)
    st.session_state['use_blur'] = st.checkbox("Apply Gaussian Blur")
    st.session_state['blur_kernel_size'] = st.slider("Blur Kernel Size", 1, 11, 3, step=2)
    
    # Enhancement
    st.subheader("Image Enhancement")
    st.session_state['use_sharpen'] = st.checkbox("Apply Sharpening", value=True)
    st.session_state['sharpen_intensity'] = st.slider("Sharpen Intensity", 0.1, 5.0, 1.5)
    st.session_state['sharpen_sigma'] = st.slider("Sharpen Sigma", 0.1, 5.0, 1.0)
    
    # Morphological operations
    st.subheader("Morphological Operations")
    st.session_state['use_morphology'] = st.checkbox("Apply Morphological Operations")
    st.session_state['morph_operation'] = st.selectbox(
        "Operation Type", 
        ["dilate", "erode", "open", "close"]
    )
    st.session_state['morph_kernel_size'] = st.slider("Kernel Size", 1, 11, 3, step=2)
    st.session_state['morph_iterations'] = st.slider("Iterations", 1, 10, 1)
    
    # Thresholding
    st.subheader("Thresholding")
    st.session_state['threshold_method'] = st.selectbox(
        "Threshold Method", 
        ["adaptive", "otsu", "binary", "none"]
    )
    st.session_state['binary_threshold'] = st.slider("Binary Threshold", 0, 255, 127)
    st.session_state['adaptive_block_size'] = st.slider("Adaptive Block Size", 3, 99, 11, step=2)
    st.session_state['adaptive_c'] = st.slider("Adaptive C", -10, 10, 2)
    
    # Edge enhancement
    st.subheader("Edge Enhancement")
    st.session_state['use_edge_enhancement'] = st.checkbox("Apply Edge Enhancement")
    st.session_state['canny_threshold1'] = st.slider("Canny Threshold 1", 0, 300, 100)
    st.session_state['canny_threshold2'] = st.slider("Canny Threshold 2", 0, 300, 200)
    
    # OCR settings
    st.subheader("OCR Settings")
    st.session_state['use_roi'] = st.checkbox("Use Region of Interest Extraction")
    st.session_state['show_rois'] = st.checkbox("Show Detected ROIs")
    
    # Only show spell check option if available
    if NLTK_AVAILABLE and SPELLCHECKER_AVAILABLE:
        st.session_state['use_spell_check'] = st.checkbox("Apply Spell Checking", value=True)
    else:
        st.warning("Spell checking disabled - NLTK or SpellChecker not available")
        st.session_state['use_spell_check'] = False
    
    st.session_state['tesseract_config'] = st.text_input(
        "Tesseract Configuration", 
        value="--psm 6 --oem 3"
    )
    
    # Add a button to restore defaults
    if st.button("Restore Default Settings", key="restore_defaults_button"):
        # Basic
        st.session_state['use_grayscale'] = True
        st.session_state['brightness'] = 10
        st.session_state['contrast'] = 1.2
        # Noise reduction
        st.session_state['use_denoise'] = True
        st.session_state['denoise_strength'] = 10
        st.session_state['use_blur'] = False
        st.session_state['blur_kernel_size'] = 3
        # Enhancement
        st.session_state['use_sharpen'] = True
        st.session_state['sharpen_intensity'] = 1.5
        st.session_state['sharpen_sigma'] = 1.0
        # Morph
        st.session_state['use_morphology'] = False
        st.session_state['morph_operation'] = 'close'
        st.session_state['morph_kernel_size'] = 3
        st.session_state['morph_iterations'] = 1
        # Threshold
        st.session_state['threshold_method'] = 'adaptive'
        st.session_state['binary_threshold'] = 127
        st.session_state['adaptive_block_size'] = 11
        st.session_state['adaptive_c'] = 2
        # Edge
        st.session_state['use_edge_enhancement'] = False
        st.session_state['canny_threshold1'] = 100
        st.session_state['canny_threshold2'] = 200
        # OCR
        st.session_state['use_roi'] = False
        st.session_state['show_rois'] = False
        st.session_state['tesseract_config'] = '--psm 6 --oem 3'
        
        st.success("Settings restored to defaults!")

with tab3:
    st.header("About Hotel Bill OCR Processor")
    st.markdown("""
    This application is designed to extract and structure information from hotel bill and menu images using advanced OCR techniques.
    
    ### Key Features
    
    * **Advanced Image Preprocessing**: Multiple techniques to optimize image quality for OCR
    * **Menu Structure Recognition**: Identifies menu categories, items, and prices
    * **Spatial Layout Analysis**: Uses the layout of text to understand menu structure
    * **Text Cleaning & Correction**: Improves OCR accuracy with NLP techniques
    
    ### How to Use
    
    1. Upload a hotel bill or menu image
    2. Adjust preprocessing settings if needed (or use defaults)
    3. Click "Process Image" to extract and structure text
    4. View the structured JSON output and download if needed
    
    ### Technologies Used
    
    * Python with Streamlit for the user interface
    * OpenCV for image processing
    * Tesseract OCR for text extraction
    * NLTK and SpellChecker for text correction (if available)
    
    ### Need Help?
    
    If you encounter any issues:
    
    * Ensure Tesseract OCR is properly installed
    * Try adjusting image preprocessing parameters to improve OCR quality
    * For menus with special formatting, try the ROI extraction option
    
    ### Version
    
    v1.0.0 - Advanced Hotel Bill OCR Processor
    """)