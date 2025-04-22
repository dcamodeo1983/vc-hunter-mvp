import os
import base64
import logging
import docx2txt
from PyPDF2 import PdfReader
import numpy as np

logger = logging.getLogger(__name__)

def load_documents_as_text(uploaded_files):
    texts = []
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        try:
            if filename.endswith(".txt"):
                texts.append(uploaded_file.read().decode("utf-8"))
            elif filename.endswith(".docx"):
                with open("/tmp/temp.docx", "wb") as f:
                    f.write(uploaded_file.read())
                text = docx2txt.process("/tmp/temp.docx")
                texts.append(text)
            elif filename.endswith(".pdf"):
                with open("/tmp/temp.pdf", "wb") as f:
                    f.write(uploaded_file.read())
                reader = PdfReader("/tmp/temp.pdf")
                text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                texts.append(text)
            else:
                texts.append(uploaded_file.read().decode("utf-8", errors="ignore"))
        except Exception as e:
            logger.warning(f"Failed to load file {filename}: {e}")
    return texts

def extract_text_from_file(file_bytes):
    try:
        # Try plain UTF-8 text first
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return convert_pdf_to_text(file_bytes)
        except Exception as e:
            logger.warning(f"Failed to extract text from PDF: {e}")
            raise ValueError("Unable to parse input as plain text or PDF.")

def convert_pdf_to_text(file_bytes):
    try:
        with open("/tmp/temp_stream.pdf", "wb") as f:
            f.write(file_bytes)
        reader = PdfReader("/tmp/temp_stream.pdf")
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        logger.error("PDF parsing failed", exc_info=True)
        raise

def safe_truncate_text(text, max_tokens, encoding_name="cl100k_base"):
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        tokens = enc.encode(text)
        return enc.decode(tokens[:max_tokens])
    except Exception as e:
        logger.warning(f"Failed to truncate text safely: {e}")
        return text[:4000]  # Fallback slice

def ensure_numpy_array(embedding):
    if isinstance(embedding, list):
        return np.array(embedding)
    elif hasattr(embedding, 'tolist'):
        return np.array(embedding.tolist())
    return embedding
