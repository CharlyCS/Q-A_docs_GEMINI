import os
from PyPDF2 import PdfReader
text = ""
dir_doc = "Q_A_docs_GEMINI/docs"
for filename in os.listdir(dir_doc):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(dir_doc, filename)
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()

output_path = os.path.join(dir_doc, "output.txt")
with open(output_path, "w", encoding="utf-8") as text_file:
    text_file.write(text)