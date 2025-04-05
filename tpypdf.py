import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path, output_txt_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file '{pdf_path}' does not exist.")
    
    extracted_text = []

    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print(f"[INFO] Total pages found: {num_pages}")

        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                # Clean up whitespace and remove empty lines
                cleaned_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
                extracted_text.append(f"--- Page {i} ---\n{cleaned_text}")
            else:
                extracted_text.append(f"--- Page {i} ---\n[No readable text found]")

        # Join all text together
        full_text = "\n\n".join(extracted_text)

        # Create output folder if it doesn't exist
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

        # Write to file
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"✅ Text extraction complete! Output saved to: {output_txt_path}")
        return full_text

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return ""

# Example usage
pdf_file = "ledger_vee.pdf"
output_file = "output/output6.txt"

text = extract_text_from_pdf(pdf_file, output_file)
