import streamlit as st
import cv2
import numpy as np
import pytesseract
import json
import re
from PIL import Image
from difflib import SequenceMatcher

# Improved preprocessing with denoising
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh

# Enhanced OCR with layout analysis
def extract_text(image):
    processed = preprocess_image(image)
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    extracted_text = pytesseract.image_to_string(processed, config=custom_config)
    return extracted_text

# Advanced item parsing with similarity checks
class MenuParser:
    def __init__(self):
        self.price_pattern = re.compile(r'(\$?\d{1,3}(?:[.,]\d{2})?)')
        self.category_keywords = {
            'main': ['main', 'entree', 'curry', 'vindaloo', 'biryani'],
            'dessert': ['dessert', 'sweet', 'ice cream', 'pastry'],
            'appetizer': ['appetizer', 'starter', 'snack'],
            'bread': ['naan', 'roti', 'bread'],
            'drinks': ['drink', 'beverage', 'juice', 'lassi']
        }
        self.seen_items = set()
        
    def similarity(self, a, b):
        return SequenceMatcher(None, a, b).ratio()
    
    def clean_item(self, text):
        # Remove price mentions from item name
        cleaned = re.sub(r'\$?\d+\.?\d*', '', text).strip()
        # Remove special characters and normalize spaces
        cleaned = re.sub(r'[^a-zA-Z ]', '', cleaned)
        return ' '.join(cleaned.split())
    
    def parse_line(self, line):
        line = line.replace('=', '').replace(':', ' ').strip()
        prices = self.price_pattern.findall(line)
        if not prices:
            return None, None
            
        # Take last valid price in line
        price = prices[-1].replace(' ', '').replace(',', '.')
        if not price.startswith('$'):
            price = f'${price}'
            
        # Extract item name before price
        price_pos = line.rfind(prices[-1])
        name = self.clean_item(line[:price_pos])
        
        return name, price
    
    def add_item(self, category, name, price):
        # Deduplicate using similarity check
        for seen in self.seen_items:
            if self.similarity(name, seen) > 0.8:
                return
        if name and price:
            category = category.lower()
            self.seen_items.add(name)
            return {"name": name.title(), "price": price}

def parse_menu(lines):
    parser = MenuParser()
    menu = {cat: [] for cat in parser.category_keywords}
    current_category = 'main'

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Category detection
        lower_line = line.lower()
        for cat, keywords in parser.category_keywords.items():
            if any(kw in lower_line for kw in keywords):
                current_category = cat
                break

        # Item processing
        name, price = parser.parse_line(line)
        if name and price:  # ✅ Ensure both name and price are not None
            item = parser.add_item(current_category, name, price)
            if item:  # ✅ Ensure a valid item was added
                menu[current_category].append(item)

    return {k: v for k, v in menu.items() if v}  # ✅ Remove empty categories


# Streamlit interface
st.title("Advanced Menu Scanner 🍔")
st.markdown("Upload a clear photo of restaurant menu")

uploaded_file = st.file_uploader("Choose image...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Scanning Menu...", use_column_width=True)
    
    extracted_text = extract_text(image)
    lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]

    # Improved restaurant info detection
    restaurant = next((line for line in lines if re.search(r'\d{2,}[\w\s,]+', line)), "Unknown Restaurant")

    # Fixed open_hours extraction with error handling
    open_hours = "Not Found"
    for line in lines:
        if re.search(r'open|hours|time', line, re.I):
            match = re.search(r'\d{1,2}[APMapm]{2}.*?\d{1,2}[APMapm]{2}', line)
            if match:
                open_hours = match.group()
                break

    menu_data = parse_menu(lines)

    result = {
        "restaurant": restaurant,
        "open_hours": open_hours,
        "menu": menu_data
    }

    st.subheader("Structured Menu Data")
    st.json(result)

    st.download_button(
        "Download JSON",
        json.dumps(result, indent=2),
        "menu.json",
        "application/json"
    )
