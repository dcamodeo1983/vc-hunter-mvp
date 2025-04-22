
from openai import OpenAI
import numpy as np
import os
from agents.utils import ensure_numpy_array, safe_truncate_text

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = os.getenv("VC_HUNTER_CHAT_MODEL", "gpt-4")
EMBED_MODEL = os.getenv("VC_HUNTER_EMBED_MODEL", "text-embedding-ada-002")

def summarize_text(text, prompt_prefix="Summarize the following VC firm or startup concept:"):
    prompt = f"{prompt_prefix}\n\n{text}\n\nSummary:"
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def embed_text(text):
    response = client.embeddings.create(
        input=[text],
        model=EMBED_MODEL
    )
    return response.data[0].embedding

def generate_founder_summary(files):
    contents = []
    for file in files:
        contents.append(file.read().decode("utf-8", errors="ignore"))
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

def match_founder_to_vcs(founder_embedding, vc_embeddings, vc_summaries):
    results = []
    for url, vc_vec in vc_embeddings.items():
        vec = ensure_numpy_array(vc_vec)
        similarity = np.dot(founder_embedding, vec.T).flatten()[0]
        narrative = generate_match_narrative(url, vc_summaries[url])
        message = generate_custom_message(url, vc_summaries[url])
        results.append({
            "vc_url": url,
            "score": round(float(similarity), 4),
            "why_match": narrative,
            "messaging_advice": message
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def generate_match_narrative(vc_url, vc_summary):
    prompt = f"A startup founder is evaluating VC firms. Here is a VC firm's summary: {vc_summary}\n\nExplain in 2–3 sentences why this firm would be a good match for a founder working in a complementary space."
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_custom_message(vc_url, vc_summary):
    prompt = f"Based on this VC firm's description, suggest a brief outreach strategy or messaging angle that a startup founder should use to connect effectively:\n\n{vc_summary}"
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def analyze_gap(vc_embeddings, founder_embedding, clusters):
    prompt = "Analyze this situation: A founder is working on a startup and here are the VC investment clusters around them. Suggest what whitespace (underfunded sectors) may exist based on the embedding clusters and the founder's concept."
    context = f"Clusters: {str(clusters)}\nFounder embedding position: {founder_embedding[:5]}"
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": f"{prompt}\n\n{context}"}]
    )
    return response.choices[0].message.content.strip()

def generate_chat_context(founder_summary, vc_summaries, top_matches):
    context = f"FOUNDER SUMMARY:\n{founder_summary}\n\n"
    context += "TOP VC MATCHES:\n"
    for match in top_matches[:3]:
        context += f"{match['vc_url']}: Score {match['score']}, Match Rationale: {match['why_match']}\n"
    context += "\nVC SUMMARIES:\n"
    for url, summary in vc_summaries.items():
        context += f"{url}: {summary}\n"
    return context
