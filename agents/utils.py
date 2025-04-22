# agents/utils.py

import os
import base64
import logging
import docx2txt
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)

def load_documents_as_text(documents):
    """Loads a list of uploaded documents and extracts their text content."""
    texts = []

    for doc in documents:
        try:
            file_type = doc.name.split(".")[-1].lower()
            logger.info(f"Processing file: {doc.name} of type {file_type}")

            if file_type == "pdf":
                reader = PdfReader(doc)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
            elif file_type == "txt":
                text = doc.read().decode("utf-8")
            elif file_type == "docx":
                with open(f"/tmp/{doc.name}", "wb") as f:
                    f.write(doc.read())
                text = docx2txt.process(f"/tmp/{doc.name}")
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                raise ValueError(f"Unsupported file type: {file_type}")

            texts.append(text.strip())
        except Exception as e:
            logger.exception(f"Failed to process document: {doc.name}")
            raise e

    return texts


def encode_plot_to_base64(fig):
    """Encodes a Matplotlib figure to a base64 string for Streamlit rendering."""
    import io
    import matplotlib.pyplot as plt

    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_bytes = buf.read()
        encoded = base64.b64encode(img_bytes).decode()
        plt.close(fig)
        logger.info("Encoded plot to base64 successfully.")
        return encoded
    except Exception as e:
        logger.exception("Failed to encode plot.")
        return None


def encode_networkx_to_base64(G):
    """Generates a network plot from a NetworkX graph and returns it as base64."""
    import io
    import matplotlib.pyplot as plt
    import networkx as nx

    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        plt.axis('off')

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_bytes = buf.read()
        encoded = base64.b64encode(img_bytes).decode()
        plt.close(fig)
        logger.info("Encoded network graph to base64 successfully.")
        return encoded
    except Exception as e:
        logger.exception("Failed to encode NetworkX graph.")
        return None
