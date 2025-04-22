
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = os.getenv("VC_HUNTER_EMBED_MODEL", "text-embedding-ada-002")

def enrich_portfolio_data(portfolio_urls):
    enriched = []
    for url in portfolio_urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string.strip() if soup.title else "Unknown Company"
            paras = soup.find_all("p")
            text = " ".join(p.get_text() for p in paras[:3])  # Limit to top 3 paragraphs

            cleaned = text.strip().replace("\n", " ")[:1000] or "No content extracted"
            embedding = generate_embedding(cleaned)
            enriched.append({
                "name": title,
                "description": cleaned,
                "embedding": embedding
            })

        except Exception:
            continue

    return enriched

def generate_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model=EMBED_MODEL
    )
    return response.data[0].embedding
