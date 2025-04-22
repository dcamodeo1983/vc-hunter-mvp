
import networkx as nx
import matplotlib.pyplot as plt

def build_relationship_graph(co_investments, competitors):
    G = nx.Graph()

    for vc1, vc2, score in co_investments:
        G.add_edge(vc1, vc2, weight=score, type="collab")

    for vc1, vc2, score in competitors:
        if G.has_edge(vc1, vc2):
            G[vc1][vc2]["type"] = "both"
        else:
            G.add_edge(vc1, vc2, weight=score, type="compete")

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
