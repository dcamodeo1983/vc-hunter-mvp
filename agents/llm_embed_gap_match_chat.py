
import os
from openai import OpenAI
from agents.utils import safe_truncate_text
import logging

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
    response = client.embeddings.create(
        input=[safe_truncate_text(text, max_tokens=7500)],
        model=EMBED_MODEL
    )
    return response.data[0].embedding

def match_founder_to_vcs(founder_embedding, vc_embeddings, vc_summaries):
    matches = []
    for vc in vc_embeddings:
        if isinstance(vc, dict) and "embedding" in vc and "url" in vc:
            score = cosine_similarity(founder_embedding, vc["embedding"])
            vc_summary = next((s['summary'] for s in vc_summaries if s['url'] == vc['url']), "No summary available.")
            matches.append({
                "vc_url": vc["url"],
                "score": round(score, 4),
                "why_match": vc_summary,
                "messaging_advice": f"Emphasize alignment with {vc_summary.split('.')[0]}."
            })
        else:
            logger.warning(f"Skipping malformed VC entry: {vc}")
    return sorted(matches, key=lambda x: x["score"], reverse=True)

def cosine_similarity(vec1, vec2):
    from numpy import dot
    from numpy.linalg import norm
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def analyze_gap(vc_data):
    return "Gap analysis is placeholder: refine based on sector embeddings and cluster density."

def generate_chat_context(founder_summary, vc_summaries, matches):
    context = f"Founder Summary:\n{founder_summary}\n\nTop VC Matches:"
    for m in matches[:3]:
        vc = next((v for v in vc_summaries if v['url'] == m['vc_url']), None)
        if vc:
            context += f"\n- {vc['url']}: {vc['summary']}"
    return context
