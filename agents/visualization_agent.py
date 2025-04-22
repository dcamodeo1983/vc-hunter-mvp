
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def generate_visuals(vc_embeddings, clusters, relationships):
    images = {}

    # 1. Plot VC embeddings colored by cluster
    try:
        urls = list(vc_embeddings.keys())
        vecs = np.array([vc_embeddings[u] for u in urls])
        labels = [clusters[u]["cluster"] for u in urls]

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(vecs[:, 0], vecs[:, 1], c=labels, cmap='viridis')
        plt.title("VC Embeddings by Cluster")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

        for i, txt in enumerate(urls):
            plt.annotate(f"VC{i+1}", (vecs[i, 0], vecs[i, 1]), fontsize=6)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        images["cluster_plot"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
    except Exception as e:
        images["cluster_plot"] = f"Error generating cluster plot: {e}"

    # 2. Relationship heatmap placeholder
    try:
        plt.figure(figsize=(6, 4))
        plt.title("VC Relationships (placeholder)")
        plt.text(0.5, 0.5, "Relationship matrix visualization", ha='center', va='center')
        plt.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        images["relationship_plot"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
    except Exception as e:
        images["relationship_plot"] = f"Error generating relationship plot: {e}"

    return images
