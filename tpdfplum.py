import pdfplumber
import os

def extract_text_from_pdf(pdf_path, output_txt_path, extract_tables=False):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file '{pdf_path}' does not exist.")

    extracted_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages found: {total_pages}")
            
            for i, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                
                if page_text:
                    cleaned_text = "\n".join(line.strip() for line in page_text.splitlines() if line.strip())
                    extracted_text.append(f"--- Page {i} ---\n{cleaned_text}")

                if extract_tables:
                    tables = page.extract_tables()
                    for table in tables:
                        table_text = "\n".join(["\t".join(row) for row in table if row])
                        extracted_text.append(f"--- Table from Page {i} ---\n{table_text}")

        full_text = "\n\n".join(extracted_text)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"✅ Extraction complete. Text saved to: {output_txt_path}")
        return full_text

    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return ""

# Example usage
pdf_file = "ledger_vee.pdf"
output_file = "output/output5.txt"  # Creates 'output' folder if needed

text = extract_text_from_pdf(pdf_file, output_file, extract_tables=True)
