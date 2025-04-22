import os
import logging
import time
from openai import OpenAI
from agents.utils import safe_truncate_text

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODEL = os.getenv("VC_HUNTER_CHAT_MODEL", "gpt-4")
EMBED_MODEL = os.getenv("VC_HUNTER_EMBED_MODEL", "text-embedding-ada-002")

MAX_INPUT_TOKENS = 8000  # conservative for 16k model

def generate_founder_summary(text):
    safe_text = safe_truncate_text(text, max_tokens=MAX_INPUT_TOKENS)
    logger.info(f"[Founder Summary] Truncated token count: ~{len(safe_text) // 4} tokens")

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": f"Summarize this founder document:\n\n{safe_text}"}]
    )
    summary = response.choices[0].message.content.strip()
    time.sleep(1)
    embed = generate_embedding(summary)
    return summary, embed

def generate_vc_summary(vc_url, scraped_text, portfolio_info):
    formatted_portfolio = ", ".join([f"{item['name']}: {item['description']}" for item in portfolio_info])
    combined = f"Website: {vc_url}\n\nDescription:\n{scraped_text}\n\nPortfolio:\n{formatted_portfolio}"
    safe_combined = safe_truncate_text(combined, max_tokens=MAX_INPUT_TOKENS)
    logger.info(f"[VC Summary] Truncated token count: ~{len(safe_combined) // 4} tokens")

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": f"Summarize this VC firm:\n\n{safe_combined}"}]
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

def cosine_similarity(vec1, vec2):
    from numpy import dot
    from numpy.linalg import norm
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

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
    joined_vcs = "\n\n".join(vc_summaries)
    combined = f"Founder Summary:\n{founder_summary}\n\nVC Summaries:\n{joined_vcs}"
    safe_combined = safe_truncate_text(combined, max_tokens=MAX_INPUT_TOKENS)
    logger.info(f"[Gap Analysis] Combined prompt truncated to ~{len(safe_combined) // 4} tokens")

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": 
                   "Given the following founder's summary and the summaries of various VCs, "
                   "analyze the whitespace or mismatch in themes or focus areas. Highlight any unmet needs or gaps.\n\n"
                   + safe_combined}]
    )
    return response.choices[0].message.content.strip()

def generate_chatbot_response(query, founder_summary, vc_summaries):
    context = generate_chat_context(founder_summary, vc_summaries, [])
    prompt = f"{context}\n\nUser question: {query}"
    safe_prompt = safe_truncate_text(prompt, max_tokens=MAX_INPUT_TOKENS)
    logger.info(f"[Chatbot] Prompt length after truncation: ~{len(safe_prompt) // 4} tokens")

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": safe_prompt}]
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
        except Exception as e:
            logger.warning(f"Failed to generate {embedding_type} embedding for: {entity} | Error: {e}")
    return results
