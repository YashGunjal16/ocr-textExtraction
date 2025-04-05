Absolutely! Here's your **copy-paste ready `README.md`** file:
# 🧾 OCR Text Extraction Tool

This tool is designed to extract structured text data from **images** and **PDF files** using OCR (Optical Character Recognition) techniques. It integrates Tesseract OCR, OpenCV for preprocessing, PDF parsers, and Streamlit for a user-friendly interface.

---

## 📚 Features

- 🔍 OCR using **Tesseract**
- 📷 Image preprocessing with **OpenCV**
- 📄 PDF parsing using **pdfplumber** and **PyPDF**
- 📊 JSON output formatting
- 🧠 Intelligent matching using **difflib**
- 🌐 Web UI with **Streamlit**
- 🧼 Regex-based text cleaning
- 🗂 File handling and directory support

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YashGunjal16/ocr-textExtraction.git
cd ocr-textExtraction
```

### 2. Install Python Dependencies

Install all necessary packages:

```bash
pip install -r requirements.txt
```

> If you prefer manual install:

```bash
pip install streamlit opencv-python numpy pytesseract Pillow pdfplumber pypdf
```

### 3. Install Tesseract-OCR

Tesseract is a required external dependency for OCR.

#### 📥 Install Instructions:

- **Windows:**  
  Download and install from [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)

- **Ubuntu/Linux:**
  ```bash
  sudo apt install tesseract-ocr
  ```

- **macOS:**
  ```bash
  brew install tesseract
  ```

#### ⚙️ Add Tesseract to PATH (Windows Only):

Add the Tesseract installation path to your system's environment variables:

```
C:\Program Files\Tesseract-OCR
```

---

## 🚀 Usage

### ▶️ Run from Command Line:

```bash
python ultimateocr.py
```

### 🌐 Run the Streamlit App:

```bash
streamlit run ultimateocr.py
```

---

## 🧪 Imports Used

Here's a complete list of Python libraries and modules used in this project:

### 🔧 Core Packages:
- `os` — File/directory operations
- `re` — Regular expressions for text cleaning
- `json` — For structured output
- `collections.defaultdict` — For structured data handling

### 🖼 Image Processing:
- `cv2` (OpenCV) — Image loading and preprocessing
- `PIL.Image` — Handling images in memory

### 🧠 OCR & Matching:
- `pytesseract` — Tesseract OCR engine
- `difflib` — Fuzzy matching (e.g., `get_close_matches`, `SequenceMatcher`)

### 📄 PDF Handling:
- `pdfplumber` — Extracting text from PDFs
- `pypdf.PdfReader` — PDF structure parsing

### 🌐 Web UI:
- `streamlit` — To create an interactive user interface

---

## 📁 Project Structure

```
.
├── bc.py
├── new.py
├── onetwo.py
├── preprocess.py
├── processed_image.png
├── tpdfplum.py
├── tpypdf.py
├── two.py
├── ultimateocr.py
├── workingversion.txt
├── requirements.txt
└── README.md
```

---

## 📌 Output

- Extracted text will be displayed or exported as structured **JSON**
- You can visualize or further process this output

---

## 👨‍💻 Author

**Yash Gunjal**  
🔗 [GitHub](https://github.com/YashGunjal16)  
💬 For questions or feedback, feel free to open an issue or drop a message!

---
