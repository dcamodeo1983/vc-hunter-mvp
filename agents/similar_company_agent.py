# agents/similar_company_agent.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_companies(founder_embedding, vc_embeddings, threshold=0.75):
    """
    Finds similar companies based on cosine similarity with the founder's embedding.

    Args:
        founder_embedding (list or np.array): The founder's semantic embedding vector.
        vc_embeddings (list of dict): Each dict contains 'url', 'embedding', and 'portfolio'.
        threshold (float): Similarity threshold to consider a match.

    Returns:
        list of dicts: Each dict contains 'name', 'description', and 'vc' firm URL.
    """
    founder_vec = np.array(founder_embedding).reshape(1, -1)
    similar = []

    for vc in vc_embeddings:
        portfolio = vc.get("portfolio", [])
        for company in portfolio:
            company_embedding = company.get("embedding")
            if company_embedding:
                sim_score = cosine_similarity(founder_vec, [company_embedding])[0][0]
                if sim_score >= threshold:
                    similar.append({
                        "name": company.get("name", "Unknown Company"),
                        "description": company.get("description", "No description available."),
                        "vc": vc.get("url", "Unknown VC")
                    })
    return similar
