import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import seaborn as sns

def generate_cluster_plot(vc_embeddings):
    urls = [vc["url"] for vc in vc_embeddings]
    vectors = np.array([vc["embedding"] for vc in vc_embeddings])
    summaries = [vc.get("summary", "No summary") for vc in vc_embeddings]
    clusters = [vc.get("cluster", 0) for vc in vc_embeddings]
    themes = [vc.get("theme", "N/A") for vc in vc_embeddings]

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    coords = tsne.fit_transform(vectors)

    df = pd.DataFrame({
        "vc_url": urls,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": clusters,
        "summary": summaries,
        "theme": themes
    })

    fig = go.Figure()

    for label, group in df.groupby("cluster"):
        if len(group) >= 3:
            points = group[["x", "y"]].values
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_points = np.append(hull_points, [hull_points[0]], axis=0)
            fig.add_trace(go.Scatter(
                x=hull_points[:, 0],
                y=hull_points[:, 1],
                fill="toself",
                mode="lines",
                line=dict(width=1),
                name=f"Cluster {label}",
                hoverinfo="text",
                text=[f"Cluster {label}: {themes[0]}"] * len(hull_points),
                opacity=0.2,
                showlegend=False
            ))

    fig.add_trace(go.Scatter(
        x=df["x"],
        y=df["y"],
        mode="markers",
        marker=dict(size=8, color=df["cluster"], colorscale="Viridis", showscale=True),
        text=df["vc_url"] + "<br><br><b>Theme:</b> " + df["theme"] + "<br><br><b>Summary:</b><br>" + df["summary"],
        hoverinfo="text",
        name="VC Firms"
    ))

    fig.update_layout(title="VC Landscape Cluster Visualization", height=600)
    return fig

def generate_heatmap_from_themes(theme_counts):
    theme_df = pd.DataFrame.from_dict(theme_counts, orient="index", columns=["count"]).sort_values("count", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(theme_df.T, cmap="YlGnBu", annot=True, fmt="d", cbar=False)
    ax.set_title("VC Investment Theme Intensity")
    plt.tight_layout()
    return fig
