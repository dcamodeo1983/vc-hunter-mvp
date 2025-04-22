
from sklearn.cluster import KMeans
import numpy as np
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def categorize_vcs(vc_embeddings, vc_summaries, n_clusters=5):
    urls = list(vc_embeddings.keys())
    vecs = np.array([vc_embeddings[url] for url in urls])
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(vecs)

    cluster_groups = {}
    for i, url in enumerate(urls):
        label = int(labels[i])
        cluster_groups.setdefault(label, []).append(url)

    cluster_descriptions = {}
    for label, urls_in_cluster in cluster_groups.items():
        summaries = [vc_summaries[url] for url in urls_in_cluster]
        combined = "\n".join(summaries)
        theme = extract_cluster_theme(combined)
        cluster_descriptions[label] = {
            "theme": theme,
            "vc_urls": urls_in_cluster
        }

    return labels.tolist(), cluster_descriptions

def extract_cluster_theme(text_block):
    prompt = f"""
You are analyzing a group of VC firms based on their summaries.

Here is a set of firm descriptions:\n\n{text_block}

Based on this, name the investment theme that most characterizes this group. Return a short label and one-sentence description.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
