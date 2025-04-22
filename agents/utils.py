import os
import base64
import logging
import docx2txt
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)

def load_documents_as_text(file_paths):
    text_blobs = []
    for file_path in file_paths:
        try:
            if file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text_blobs.append(f.read())
            elif file_path.endswith(".docx"):
                text_blobs.append(docx2txt.process(file_path))
            elif file_path.endswith(".pdf"):
                reader = PdfReader(file_path)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                text_blobs.append(text)
            else:
                logger.warning(f"Unsupported file format: {file_path}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
    return text_blobs

def encode_file_to_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode file {file_path} to base64: {e}")
        return None

def safe_truncate_text(text, max_tokens, encoding_name="cl100k_base"):
    try:
