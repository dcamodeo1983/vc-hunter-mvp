
import numpy as np
from agents.utils import ensure_numpy_array

def find_similar_portfolio_companies(founder_embedding, enriched_data):
    # For simplicity, assign random scores to simulate similarity
    similar = []
    for vc, companies in enriched_data.items():
        for company in companies:
            sim_score = np.round(np.random.uniform(0.7, 0.95), 4)  # simulate similarity
            similar.append({
                "company_name": company,
                "vc_url": vc,
                "similarity": sim_score
            })

    similar.sort(key=lambda x: x["similarity"], reverse=True)
    return similar[:5]  # return top 5 most similar
