import io
import docx
import PyPDF2
from typing import List

def extract_text(file_content: bytes, filename: str) -> str:
    try:
        if filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        elif filename.endswith('.docx'):
            doc = docx.Document(io.BytesIO(file_content))
            return "\n".join([para.text for para in doc.paragraphs])
        elif filename.endswith('.txt'):
            return file_content.decode('utf-8')
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""
    return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks