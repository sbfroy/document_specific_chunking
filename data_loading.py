from pypdf import PdfReader


def read_pdf(path_to_pdf):
    pdf_text = ''
    reader = PdfReader(path_to_pdf)
    for page in range(len(reader.pages)):
        text = reader.pages[page].extract_text()
        pdf_text += text
    return pdf_text
