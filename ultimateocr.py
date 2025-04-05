import streamlit as st
import pytesseract
import cv2
import numpy as np
import json
import re
from PIL import Image

# Set Tesseract command (adjust to your install path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def advanced_preprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize based on DPI
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bilateral filtering for noise reduction but edge preservation
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, h=30)

    return denoised

def extract_text(image, use_lstm):
    config = '--psm 6'
    config += ' --oem 1' if use_lstm else ' --oem 3'  # LSTM only or default
    text = pytesseract.image_to_string(image, config=config)
    return text

def parse_menu(text):
    menu = {}
    lines = text.split('\n')
    current_category = "Uncategorized"

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Category detection: line with no digits and maybe all caps
        if re.match(r'^[A-Za-z\s&]+$', line) and len(line) > 2 and line.upper() == line:
            current_category = line.strip()
            menu[current_category] = []
            continue

        # Item & price detection (e.g., Chicken Curry ........ 180)
        match = re.match(r'^(.*?)\s*[\.\-]*\s*₹?\s?(\d+(?:\.\d{1,2})?)$', line)
        if match:
            item, price = match.groups()
            if current_category not in menu:
                menu[current_category] = []
            menu[current_category].append({'item': item.strip(), 'price': price.strip()})
        else:
            # If no price, just add the item
            if current_category not in menu:
                menu[current_category] = []
            menu[current_category].append({'item': line.strip(), 'price': 'N/A'})

    return menu

def main():
    st.title("🍽️ Hotel Menu OCR Extractor")
    st.markdown("Upload a hotel menu image and extract structured JSON output.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    use_lstm = st.checkbox("Use LSTM OCR Engine (Tesseract --oem 1)", value=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        st.image(image, caption="Uploaded Menu", use_column_width=True)

        st.subheader("🔍 Preprocessing Image...")
        preprocessed = advanced_preprocess(img_np)
        st.image(preprocessed, caption="Preprocessed Image", channels="GRAY", use_column_width=True)

        st.subheader("🧠 Extracting Text with OCR...")
        extracted_text = extract_text(preprocessed, use_lstm)
        st.text_area("Extracted Text", extracted_text, height=200)

        st.subheader("📦 Parsing Text into JSON...")
        menu_json = parse_menu(extracted_text)
        st.json(menu_json)

        st.download_button("Download JSON", json.dumps(menu_json, indent=4), "menu.json", "application/json")

if __name__ == "__main__":
    main()
