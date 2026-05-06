import fitz
import sys

pdf_path = "data/health/AB_PM_JAY_Compendium_of_Best_Practices_2025_26_e90f2f60cb.pdf"
doc = fitz.open(pdf_path)
new_doc = fitz.open()
new_doc.insert_pdf(doc, from_page=5, to_page=5)
pdf_bytes = new_doc.write()
print(f"Original pages: {len(doc)}")
print(f"1-page PDF size: {len(pdf_bytes)} bytes")
