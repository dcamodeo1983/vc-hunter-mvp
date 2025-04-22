
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def categorize_vcs(vc_embeddings):
    vc_urls = list(vc_embeddings.keys())
    vectors = np.array([vec for vec in vc_embeddings.values()])
    best_k = 2
    best_score = -1
    best_labels = []

    for k in range(2, min(10, len(vectors))):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(vectors)
        score = silhouette_score(vectors, labels)
        if score > best_score:
            best_k = k
            best_score = score
            best_labels = labels

    result = {}
    for i, url in enumerate(vc_urls):
        result[url] = {"cluster": int(best_labels[i])}

    return result
