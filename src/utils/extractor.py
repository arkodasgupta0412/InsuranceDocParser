from PyPDF2 import PdfReader
import docx2txt
from io import BytesIO
from langchain.schema import Document
import os
from email import policy
from email.parser import BytesParser
import extract_msg
import tempfile

def load_pdf_from_bytes(pdf_bytes: bytes) -> list:
    reader = PdfReader(BytesIO(pdf_bytes))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            docs.append(Document(page_content=text, metadata={"page": i + 1}))

    return docs


def load_docx_from_bytes(docx_bytes: bytes) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
        temp_file.write(docx_bytes)
        temp_path = temp_file.name
    try:
        full_text = docx2txt.process(temp_path)
    finally:
        os.remove(temp_path)

    return [Document(page_content=full_text.strip(), metadata={"source": "docx"})]


def load_email_from_eml_bytes(eml_bytes: bytes) -> list:
    msg = BytesParser(policy=policy.default).parsebytes(eml_bytes)
    subject = msg['subject'] or ''
    body = msg.get_body(preferencelist=('plain', 'html'))
    body_text = body.get_content() if body else ''
    full_text = f"Subject: {subject}\n\n{body_text.strip()}"
    return [Document(page_content=full_text, metadata={"source": "email", "format": "eml"})]


def load_email_from_msg_bytes(msg_bytes: bytes) -> list:
    with tempfile.NamedTemporaryFile(suffix=".msg", delete=False) as temp_file:
        temp_file.write(msg_bytes)
        temp_file_path = temp_file.name
    msg = extract_msg.Message(temp_file_path)
    msg_sender = msg.sender or ''
    msg_subject = msg.subject or ''
    msg_body = msg.body or ''
    os.remove(temp_file_path)
    full_text = f"From: {msg_sender}\nSubject: {msg_subject}\n\n{msg_body.strip()}"

    return [Document(page_content=full_text, metadata={"source": "email", "format": "msg"})]


def load_document_from_bytes(content: bytes, file_type: str) -> list:
    if file_type == "PDF":
        return load_pdf_from_bytes(content)
    
    elif file_type == "DOCX":
        return load_docx_from_bytes(content)
    
    elif file_type == "EML":
        return load_email_from_eml_bytes(content)
    
    elif file_type == "MSG":
        return load_email_from_msg_bytes(content)
    
    else:
        return [Document(page_content=content.decode("utf-8")[:], metadata={"source": "text"})]