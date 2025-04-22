
# llm_embed_gap_match_chat.py

import os
import logging
import numpy as np
from openai import OpenAI
from agents.utils import safe_truncate_text

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODEL = os.getenv("VC_HUNTER_CHAT_MODEL", "gpt-4")
EMBED_MODEL = os.getenv("VC_HUNTER_EMBED_MODEL", "text-embedding-ada-002")


def generate_founder_summary(text):
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": f"Summarize this founder document: {text}"}]
    )
    summary = response.choices[0].message.content.strip()
    embed = generate_embedding(summary)
    return summary, embed


def generate_vc_summary(vc_url, scraped_text, portfolio_info):
    formatted_portfolio = ", ".join([f"{item['name']}: {item['description']}" for item in portfolio_info])
    combined = f"Website: {vc_url}\n\nDescription:\n{scraped_text}\n\nPortfolio:\n{formatted_portfolio}"
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": f"Summarize this VC firm: {combined}"}]
    )
    summary = response.choices[0].message.content.strip()
    embed = generate_embedding(summary)
    return summary, embed


def generate_embedding(text):
    truncated_text = safe_truncate_text(text, max_tokens=7500)
    response = client.embeddings.create(
        input=[truncated_text],
        model=EMBED_MODEL
    )
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def match_founder_to_vcs(founder_embedding, vc_embeddings, vc_summaries):
    matches = []
    for vc in vc_embeddings:
        if isinstance(vc, dict) and "embedding" in vc and "url" in vc:
            score = cosine_similarity(founder_embedding, vc["embedding"])
            vc_summary = next((s['summary'] for s in vc_summaries if s['url'] == vc['url']), "No summary available.")
            matches.append({
                "vc_url": vc["url"],
                "vc_name": vc.get("name", vc["url"]),
                "score": round(score, 4),
                "match_reason": vc_summary,
                "messaging_advice": f"Emphasize alignment with {vc_summary.split('.')[0]}."
            })
        else:
            logger.warning(f"Skipping malformed VC entry: {vc}")
    return sorted(matches, key=lambda x: x["score"], reverse=True)


def analyze_gap(founder_summary, vc_summaries):
    prompt = (
        "Given the following founder's summary and the summaries of various VCs, "
        "analyze the whitespace or mismatch in themes or focus areas. Highlight any unmet needs or gaps.\n\n"
        f"Founder Summary:\n{founder_summary}\n\n"
        f"VC Summaries:\n" + "\n\n".join(vc_summaries)
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def generate_chatbot_response(query, founder_summary, vc_summaries):
    context = f"Founder Summary:\n{founder_summary}\n\n"
    context += "VC Summaries:\n" + "\n\n".join([f"{v['url']}:\n{v['summary']}" for v in vc_summaries])
    prompt = f"{context}\n\nUser Question:\n{query}"
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

