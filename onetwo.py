import streamlit as st
import cv2
import numpy as np
import pytesseract
import json
import re
from PIL import Image
from difflib import get_close_matches

def preprocess_image(image):
    # Make a copy to avoid modifying the original
    img = image.copy()
    
    # Convert to grayscale if it's a color image
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Resize to a standard size while maintaining aspect ratio
    height, width = gray.shape
    target_width = 2000
    if width > target_width:
        ratio = target_width / width
        new_height = int(height * ratio)
        gray = cv2.resize(gray, (target_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 21, 10
    )
    
    # Invert if needed (check if more white than black)
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Apply dilation to make text more prominent
    dilated = cv2.dilate(opening, kernel, iterations=1)
    
    # Apply erosion to remove small noise
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    return eroded

def extract_text_from_regions(image):
    """Extract text from different regions of the image separately"""
    height, width = image.shape[:2]
    
    # Define regions (top, middle, bottom)
    top_region = image[0:int(height*0.25), :]
    middle_region = image[int(height*0.25):int(height*0.75), :]
    bottom_region = image[int(height*0.75):, :]
    
    # Process each region
    top_processed = preprocess_image(top_region)
    middle_processed = preprocess_image(middle_region)
    bottom_processed = preprocess_image(bottom_region)
    
    # Extract text from each region with appropriate PSM modes
    custom_config_title = r'--oem 3 --psm 4 -l eng --dpi 300'
    custom_config_content = r'--oem 3 --psm 6 -l eng --dpi 300'
    
    top_text = pytesseract.image_to_string(top_processed, config=custom_config_title)
    middle_text = pytesseract.image_to_string(middle_processed, config=custom_config_content)
    bottom_text = pytesseract.image_to_string(bottom_processed, config=custom_config_content)
    
    # Combine texts
    full_text = top_text + "\n" + middle_text + "\n" + bottom_text
    
    return full_text

def extract_text_with_multiple_methods(image):
    """Try multiple preprocessing methods and combine results"""
    # Method 1: Standard preprocessing
    processed1 = preprocess_image(image)
    
    # Method 2: Alternative preprocessing with different parameters
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, processed2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 3: CLAHE preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed3 = clahe.apply(gray)
    _, processed3 = cv2.threshold(processed3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Extract text with different configurations
    config1 = r'--oem 3 --psm 6 -l eng --dpi 300'
    config2 = r'--oem 3 --psm 4 -l eng --dpi 300'
    config3 = r'--oem 3 --psm 11 -l eng --dpi 300'
    
    text1 = pytesseract.image_to_string(processed1, config=config1)
    text2 = pytesseract.image_to_string(processed2, config=config2)
    text3 = pytesseract.image_to_string(processed3, config=config3)
    
    # Combine texts (will be processed and deduplicated later)
    combined_text = text1 + "\n" + text2 + "\n" + text3
    
    return combined_text

def extract_text(image):
    """Main text extraction function combining multiple approaches"""
    # Try region-based extraction
    region_text = extract_text_from_regions(image)
    
    # Try multiple method extraction
    multi_method_text = extract_text_with_multiple_methods(image)
    
    # Combine results
    combined_text = region_text + "\n" + multi_method_text
    
    # Save processed images for debugging
    processed = preprocess_image(image)
    cv2.imwrite("processed_image.png", processed)
    
    return combined_text

class MenuParser:
    def __init__(self):
        # Common menu items for correction
        self.common_items = {
            'main': ['Butter Chicken', 'Palak Paneer', 'Spicy Pork Vindaloo', 'Jerk Chicken', 
                    'Chicken Curry', 'Beef Steak', 'Grilled Salmon', 'Pasta Carbonara'],
            'dessert': ['Ice Cream', 'Chocolate Cake', 'Cheesecake', 'Apple Pie', 'Tiramisu'],
            'drink': ['Coca Cola', 'Pepsi', 'Orange Juice', 'Coffee', 'Tea', 'Wine', 'Beer'],
            'additional': ['French Fries', 'Garlic Bread', 'Side Salad', 'Rice', 'Naan']
        }
        
        # Price pattern with more variations
        self.price_pattern = re.compile(r'(\$\s*\d+\.?\d*|\d+\.?\d{2})\s*$')
        
        # Category keywords
        self.categories = {
            'main': ['main', 'course', 'curry', 'special', 'food', 'entree', 'main food'],
            'dessert': ['dessert', 'sweet', 'ice cream', 'cake', 'pudding'],
            'drink': ['drink', 'beverage', 'juice', 'wine', 'cocktail', 'beer', 'coffee', 'tea'],
            'additional': ['extra', 'side', 'additional', 'add on', 'supplement']
        }
        
        self.current_category = 'main'
        self.menu = {cat: [] for cat in self.categories}
        self.last_category_line = ""
        self.processed_lines = set()  # To avoid duplicates

    def correct_text(self, text, category=None):
        """Correct common OCR errors using fuzzy matching"""
        if not text or len(text) < 3:
            return text
            
        # Basic corrections
        # corrections = {
        #     'chichen': 'chicken', 'chieken': 'chicken', 'chickeh': 'chicken',
        #     'paneer': 'paneer', 'panner': 'paneer', 'paneor': 'paneer',
        #     'vinorlad': 'vindaloo', 'vindalo': 'vindaloo', 'vindaloc': 'vindaloo',
        #     'spicy': 'spicy', 'spicey': 'spicy', 'spicv': 'spicy',
        #     'purten': 'butter', 'buter': 'butter', 'buttor': 'butter'
        # }
        
        # Apply basic corrections
        words = text.lower().split()
        corrected_words = []
        
        for word in words:
            if word in corrections:
                corrected_words.append(corrections[word])
            else:
                corrected_words.append(word)
        
        corrected_text = ' '.join(corrected_words)
        
        # Try fuzzy matching with common items if category is known
        if category and category in self.common_items:
            matches = get_close_matches(corrected_text.title(), self.common_items[category], n=1, cutoff=0.6)
            if matches:
                return matches[0]
        
        return corrected_text.title()

    def clean_name(self, text):
        """Clean and normalize item names"""
        # Remove price remnants and special characters
        cleaned = re.sub(r'[\$\d\.]+', '', text)
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', cleaned)
        
        # Remove common filler words
        filler_words = ['and', 'with', 'the', 'our', 'for', 'from', 'of', 'to']
        words = [word for word in cleaned.strip().split() if len(word) > 1 and word.lower() not in filler_words]
        
        if not words:
            return None
            
        # Take first 3-4 meaningful words for the name
        name = ' '.join(words[:4]).title()
        
        # Apply text correction
        corrected_name = self.correct_text(name, self.current_category)
        
        return corrected_name

    def detect_category(self, line):
        """Detect menu category from a line of text"""
        line_lower = line.lower().strip()
        
        # Check if line is a category header
        if len(line) < 30 and (line.isupper() or any(line_lower.startswith(cat) for cat in self.categories)):
            for cat, keywords in self.categories.items():
                if any(kw in line_lower for kw in keywords):
                    self.last_category_line = line
                    return cat
        return None

    def parse_line(self, line):
        """Parse a line of text to extract item name and price"""
        line = line.strip()
        if not line or len(line) < 3:
            return None, None

        # Skip already processed lines
        line_hash = hash(line)
        if line_hash in self.processed_lines:
            return None, None
        self.processed_lines.add(line_hash)

        # Skip category headers and very short lines
        if line.lower() == self.last_category_line.lower() or len(line) < 5:
            return None, None

        # Category detection
        if (category := self.detect_category(line)):
            self.current_category = category
            return None, None

        # Price extraction
        price_match = re.search(self.price_pattern, line)
        if not price_match:
            # Try to find price in different format
            price_match = re.search(r'(\d+\.?\d*)', line)
            if not price_match or float(price_match.group(1)) < 1:  # Avoid capturing non-price numbers
                return None, None
        
        price = price_match.group(1)
        # Clean up price format
        price = price.replace(' ', '').strip()
        if not price.startswith('$'):
            price = '$' + price
        if '.' not in price:
            price += ".00"
        
        try:
            price_value = float(price.replace('$', ''))
            price = f"${price_value:.2f}"
        except ValueError:
            return None, None

        # Name cleaning - extract text before the price
        name_part = line[:price_match.start()].strip()
        name = self.clean_name(name_part)
        
        return name, price

    def add_item(self, name, price):
        """Add an item to the menu if it's not already there"""
        if name and price and not any(item['name'] == name for item in self.menu[self.current_category]):
            self.menu[self.current_category].append({
                "name": name,
                "price": price
            })

def parse_menu(text):
    """Parse the extracted text to build a structured menu"""
    parser = MenuParser()
    
    # Split text into lines and remove duplicates
    lines = []
    seen = set()
    for line in text.split('\n'):
        line = line.strip()
        if line and line not in seen:
            lines.append(line)
            seen.add(line)
    
    # First pass - detect categories
    for line in lines:
        parser.detect_category(line)
    
    # Second pass - extract items
    for line in lines:
        name, price = parser.parse_line(line)
        if name and price:
            parser.add_item(name, price)
    
    # Clean up empty categories
    return {k: v for k, v in parser.menu.items() if v}

def detect_restaurant_name(text, image):
    """Detect the restaurant name from the image and text"""
    # Process the top portion of the image specifically for the restaurant name
    height, width = image.shape[:2]
    top_portion = image[0:int(height/6), :]  # Top sixth of image
    
    # Apply specific preprocessing for title text
    gray = cv2.cvtColor(top_portion, cv2.COLOR_BGR2GRAY) if len(top_portion.shape) == 3 else top_portion
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Try multiple PSM modes for better title extraction
    configs = [
        r'--oem 3 --psm 4 -l eng',  # Assume single column of text
        r'--oem 3 --psm 11 -l eng',  # Sparse text. Find as much text as possible in no particular order
        r'--oem 3 --psm 3 -l eng'    # Fully automatic page segmentation, but no OSD
    ]
    
    potential_names = []
    
    for config in configs:
        top_text = pytesseract.image_to_string(binary, config=config)
        lines = [line.strip() for line in top_text.split('\n') if len(line.strip()) > 2]
        
        for line in lines:
            # Look for potential restaurant name indicators
            if (len(line) > 3 and len(line) < 30 and 
                not re.search(r'[\d\$]', line) and 
                not any(kw in line.lower() for kw in ['menu', 'food', 'drink', 'dessert'])):
                potential_names.append(line)
    
    # Also check the first few lines of the full extracted text
    text_lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 2]
    for line in text_lines[:10]:
        if (len(line) > 3 and len(line) < 30 and 
            not re.search(r'[\d\$]', line) and 
            not any(kw in line.lower() for kw in ['menu', 'food', 'drink', 'dessert'])):
            potential_names.append(line)
    
    # Clean up potential names
    cleaned_names = []
    for name in potential_names:
        # Remove special characters and normalize
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', name)
        cleaned = ' '.join(cleaned.split())  # Normalize whitespace
        if cleaned and len(cleaned) > 2:
            cleaned_names.append(cleaned.title())
    
    # Return the most common name or the first one
    if cleaned_names:
        from collections import Counter
        name_counts = Counter(cleaned_names)
        most_common = name_counts.most_common(1)[0][0]
        return most_common
    
    return "HOTEL NAME"  # Default fallback

# Streamlit interface
st.title("Professional Menu Scanner")
st.write("Upload a menu image to extract items and prices")

uploaded_file = st.file_uploader("Upload Menu Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load and display image
    pil_image = Image.open(uploaded_file)
    image = np.array(pil_image)
    st.image(image, caption="Uploaded Menu", use_column_width=True)
    
    # Process image with progress indicator
    with st.spinner("Processing image..."):
        # Extract text with enhanced methods
        extracted_text = extract_text(image)
        
        # Show extracted text in expandable section
        with st.expander("View Raw Extracted Text"):
            st.text(extracted_text)
        
        # Detect restaurant name
        restaurant = detect_restaurant_name(extracted_text, image)
        
        # Parse menu items
        menu_data = parse_menu(extracted_text)
        
        # Prepare result
        result = {
            "restaurant": restaurant,
            "open_hours": "Not Found",
            "menu": menu_data
        }
        
        # Display results
        st.subheader("Extracted Menu Data")
        st.json(result)
        
        # Download option
        st.download_button(
            "Download JSON",
            json.dumps(result, indent=2),
            "menu.json",
            "application/json"
        )
        
        # Display statistics
        total_items = sum(len(items) for items in menu_data.values())
        st.success(f"Successfully extracted {total_items} menu items across {len(menu_data)} categories.")