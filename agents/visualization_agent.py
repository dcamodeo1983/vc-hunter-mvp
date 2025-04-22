import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import seaborn as sns

def generate_tsne_plot(vc_embeddings, clusters, vc_summaries, cluster_descriptions):
    urls = list(vc_embeddings.keys())
    vectors = np.array([vc_embeddings[url] for url in urls])
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    coords = tsne.fit_transform(vectors)

    df = pd.DataFrame({
        "vc_url": urls,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": clusters,
        "summary": [vc_summaries[u] for u in urls],
        "theme": [cluster_descriptions[c]["theme"] for c in clusters]
    })

    fig = go.Figure()

    # Draw cluster hulls
    for label, group in df.groupby("cluster"):
        if len(group) >= 3:
            points = group[["x", "y"]].values
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_points = np.append(hull_points, [hull_points[0]], axis=0)  # Close the loop
            fig.add_trace(go.Scatter(
                x=hull_points[:, 0],
                y=hull_points[:, 1],
                fill="toself",
                mode="lines",
                line=dict(width=1),
                name=f"Cluster {label}",
                hoverinfo="text",
                text=[f"Cluster {label}: {cluster_descriptions[label]['theme']}"] * len(hull_points),
                opacity=0.2,
                showlegend=False
            ))

    # Plot VCs
    fig.add_trace(go.Scatter(
        x=df["x"],
        y=df["y"],
        mode="markers",
        marker=dict(size=8, color=df["cluster"], colorscale="Vir
