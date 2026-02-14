from docling.document_converter import DocumentConverter

source = r"C:\Users\h.ayed\Downloads\CELEX_32021R0782_FR_TXT.pdf"
output_md = r"C:\Users\h.ayed\Downloads\CELEX_32021R0782_FR_TXT.md"

converter = DocumentConverter()
result = converter.convert(source)

markdown = result.document.export_to_markdown()

with open(output_md, "w", encoding="utf-8") as f:
    f.write(markdown)
