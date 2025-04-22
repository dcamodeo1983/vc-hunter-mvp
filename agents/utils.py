
import os
import json
import logging
import tiktoken
import numpy as np

logger = logging.getLogger(__name__)

def load_or_generate_embedding(text, embed_fn, cache_path):
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached embedding: {e}")

    embedding = embed_fn(text)
    with open(cache_path, "w") as f:
        json.dump(embedding, f)
    return embedding

def save_results_to_cache(results, filename="results_cache.json"):
    with open(filename, "w") as f:
        json.dump(results, f)

def load_results_from_cache(filename="results_cache.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return None

def get_vc_urls_from_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError("vc_urls.txt not found.")
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def safe_truncate_text(text, model="gpt-4", max_tokens=8000):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)

def ensure_numpy_array(data):
    if isinstance(data, list):
        return np.array(data).reshape(1, -1)
    if isinstance(data, np.ndarray):
        return data.reshape(1, -1)
    raise ValueError("Input must be list or numpy array")
