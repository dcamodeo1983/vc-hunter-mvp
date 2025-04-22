
from openai import OpenAI
import numpy as np
import os
import mimetypes
from agents.utils import ensure_numpy_array, safe_truncate_text

client = OpenAI()

# Configurable models
CHAT_MODEL = os.getenv("VC_HUNTER_CHAT_MODEL", "gpt-4")
EMBED_MODEL = os.getenv("VC_HUNTER_EMBED_MODEL", "text-embedding-ada-002")

def generate_founder_summary(files):
    contents = []
    for file in files:
        mime_type, _ = mimetypes.guess_type(file.name)
        if mime_type == "application/pdf":
            contents.append("[PDF parsing not yet implemented]")
        elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            contents.append("[DOCX parsing not yet implemented]")
        else:
            try:
                contents.append(file.read().decode("utf-8", errors="ignore"))
            except Exception as e:
                contents.append(f"[Error decoding file {file.name}: {e}]")

    combined = "\n".join(contents)
    truncated = safe_truncate_text(combined)
    summary = summarize_text(truncated)
    embedding = embed_text(summary)
    return summary, embedding

def generate_vc_summary(vc_url, scraped_text, portfolio_info):
    combined = f"Website: {vc_url}\n\nDescription:\n{scraped_text}\n\nPortfolio:\n" + ", ".join(portfolio_info)
    truncated = safe_truncate_text(combined)
    summary = summarize_text(truncated)
    embedding = embed_text(summary)
    return summary, embedding

def summarize_text(text):
    prompt = f"Summarize the following VC firm or startup concept:\n\n{text}\n\nSummary:"
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def embed_text(text):
    response = client.embeddings.create(
        input=[text],
        model=EMBED_MODEL
    )
    return response.data[0].embedding

def match_founder_to_vcs(founder_embedding, vc_embeddings):
    results = []
    for url, vc_vec in vc_embeddings.items():
        vec = ensure_numpy_array(vc_vec)
        similarity = np.dot(founder_embedding, vec.T).flatten()[0]
        results.append({"vc_url": url, "score": round(float(similarity), 4)})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def analyze_gap(vc_embeddings, founder_embedding, clusters):
    return "Sectors adjacent to your idea are well-funded, but direct competitors are sparse—suggesting a whitespace opportunity."

def generate_chat_context(founder_summary, vc_summaries):
    context = "Founder Summary:\n" + founder_summary + "\n\n"
    for url, summary in vc_summaries.items():
        context += f"VC: {url}\nSummary: {summary}\n\n"
    return context
