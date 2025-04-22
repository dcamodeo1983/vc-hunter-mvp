
from openai import OpenAI
import numpy as np
import os
from agents.utils import ensure_numpy_array

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def find_similar_portfolio_companies(founder_embedding, portfolio_embeddings):
    results = []
    for vc_url, companies in portfolio_embeddings.items():
        for company in companies:
            vec = ensure_numpy_array(company["embedding"])
            similarity = np.dot(founder_embedding, vec.T).flatten()[0]
            enriched_info = enrich_company_description(company["name"], company.get("description", ""))
            results.append({
                "company_name": company["name"],
                "vc_url": vc_url,
                "similarity": round(float(similarity), 4),
                "description": enriched_info["description"],
                "strategic_insight": enriched_info["strategic"]
            })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:5]

def enrich_company_description(name, snippet):
    prompt = f"""
You are an analyst helping founders understand their competitive landscape.

Startup Name: {name}
What it does: {snippet}

1. Describe what this startup does in 2–3 sentences.
2. Explain why it might be considered a competitor or relevant comparison for a new startup in a similar space.
3. Suggest how a founder could differentiate themselves or gain a strategic edge.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content.strip()

    parts = response.split("\n")
    return {
        "description": parts[0].strip() if parts else snippet,
        "strategic": "\n".join(parts[1:]).strip() if len(parts) > 1 else "N/A"
    }
