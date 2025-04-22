# agents/relationship_agent.py

import networkx as nx
import matplotlib.pyplot as plt

def build_relationship_graph(vc_embeddings, competitors):
    G = nx.Graph()

    # Add all VCs to the graph
    for vc in vc_embeddings:
        G.add_node(vc["url"])

    # Add co-investments as green edges
    for entry in competitors:
        vc1 = entry.get("vc1") or entry.get("vc_a") or entry.get("url_a")
        vc2 = entry.get("vc2") or entry.get("vc_b") or entry.get("url_b")
        score = entry.get("score", 1.0)
        if vc1 and vc2:
            if G.has_edge(vc1, vc2):
                G[vc1][vc2]["type"] = "both"
            else:
                G.add_edge(vc1, vc2, weight=score, type="compete")  # Default to "compete"

    return G

def plot_relationship_graph(G):
    pos = nx.spring_layout(G, seed=42, k=0.3)
    edge_colors = []
    edge_weights = []

    for u, v, data in G.edges(data=True):
        if data["type"] == "collab":
            edge_colors.append("green")
        elif data["type"] == "compete":
            edge_colors.append("red")
        else:
            edge_colors.append("purple")
        edge_weights.append(data["weight"])

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=800)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_weights)
    nx.draw_networkx_labels(G, pos, font_size=10)

    edge_labels = {(u, v): f"{d['type'][0].upper()}:{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("VC Co-Investment & Competition Network")
    plt.axis("off")
    plt.tight_layout()
    return plt
