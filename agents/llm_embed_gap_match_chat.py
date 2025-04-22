import os
import time
import logging
from openai import OpenAI
from agents.utils import safe_truncate_text
import numpy as np

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Use GPT-3.5 for cost-effective summarization & chat
CHAT_MODEL = os.getenv("VC_HUNTER_CHAT_MODEL", "gpt-3.5-turbo")
EMBED_MODEL = os.getenv("VC_HUNTER_EMBED_MODEL", "text-embedding-ada-002")
GPT4_MODEL = os.getenv("VC_HUNTER_GPT4_MODEL", "gpt-4")  # for gap analysis

def generate_founder_summary(text):
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": f"Summarize this founder document: {text}"}]
    )
    summary = response.choices[0].message.content.strip()
    time.sleep(2)
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
    time.sleep(2)
    embed = generate_embedding(summary)
    return summary, embed

def generate_embedding(text):
    response = client.embeddings.create(
        input=[safe_truncate_text(text, max_tokens=7500)],
        model=EMBED_MODEL
    )
    time.sleep(2)
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

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

def analyze_gap(founder_summary, vc_summaries):
    prompt = (
        "Given the following founder's summary and the summaries of various VCs, "
        "analyze the whitespace or mismatch in themes or focus areas. Highlight any unmet needs or gaps.\n\n"
        f"Founder Summary:\n{founder_summary}\n\n"
        f"VC Summaries:\n" + "\n\n".join(vc_summaries)
    )
    response = client.chat.completions.create(
        model=GPT4_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_chatbot_response(query, founder_summary, vc_summaries):
    context = generate_chat_context(founder_summary, vc_summaries, [])
    prompt = f"{context}\n\nUser question: {query}"
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_chat_context(founder_summary, vc_summaries, matches):
    context = f"Founder Summary:\n{founder_summary}\n\nTop VC Matches:"
    for m in matches[:3]:
        vc = next((v for v in vc_summaries if v['url'] == m['vc_url']), None)
        if vc:
            context += f"\n- {vc['url']}: {vc['summary']}"
    return context

def load_or_generate_embeddings(entities, embedding_type, generate_func):
    results = []
    for entity in entities:
        try:
            summary, embedding = generate_func(**entity) if isinstance(entity, dict) else generate_func(entity)
            results.append({
                "summary": summary,
                "embedding": embedding,
                "url": entity.get("url") if isinstance(entity, dict) else "unknown"
            })
            time.sleep(2)
        except Exception as e:
            logger.warning(f"Failed to generate {embedding_type} embedding for: {entity} | Error: {e}")
    return results
