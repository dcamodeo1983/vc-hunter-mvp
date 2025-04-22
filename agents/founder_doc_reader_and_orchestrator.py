
import os
from agents.website_scraper_agent import scrape_vc_website
from agents.portfolio_enricher_agent import enrich_portfolio_data
from agents.llm_embed_gap_match_chat import (
    generate_founder_summary,
    generate_vc_summary,
    match_founder_to_vcs,
    analyze_gap,
    generate_chat_context
)
from agents.similar_company_agent import find_similar_portfolio_companies
from agents.categorizer_agent import categorize_vcs
from agents.relationship_agent import build_relationship_graph
from agents.visualization_agent import generate_tsne_plot, generate_heatmap_from_themes
from agents.chat_agent import answer_question

def run_orchestration(founder_docs, vc_urls):
    results = {}

    # Founder embedding
    founder_summary, founder_embedding = generate_founder_summary(founder_docs)
    results["founder_summary"] = founder_summary

    vc_embeddings = {}
    vc_summaries = {}
    portfolio_embeddings = {}

    for url in vc_urls:
        text, portfolio = scrape_vc_website(url)
        enriched_portfolio = enrich_portfolio_data(portfolio)
        summary, embedding = generate_vc_summary(url, text, enriched_portfolio)
        vc_embeddings[url] = embedding
        vc_summaries[url] = summary
        portfolio_embeddings[url] = enriched_portfolio

    results["vc_summaries"] = vc_summaries

    # Matching
    matches = match_founder_to_vcs(founder_embedding, vc_embeddings, vc_summaries)
    results["matches"] = matches

    # Chat context
    chat_context = generate_chat_context(founder_summary, vc_summaries, matches)
    results["chat_context"] = chat_context

    # Similar companies
    similar = find_similar_portfolio_companies(founder_embedding, portfolio_embeddings)
    results["similar_companies"] = similar

    # Clustering
    clusters, cluster_meta = categorize_vcs(vc_embeddings, vc_summaries)
    results["clusters"] = clusters

    # Visuals
    results["visuals"] = {
        "tsne": generate_tsne_plot(vc_embeddings, clusters, vc_summaries, cluster_meta),
        "heatmap": generate_heatmap_from_themes(
            {meta["theme"]: len(meta["vc_urls"]) for meta in cluster_meta.values()}
        )
    }

    # Relationships
    co_investments = []
    competitors = []
    for sim in similar:
        co_investments.append((sim["vc_url"], matches[0]["vc_url"], sim["similarity"]))
        competitors.append((matches[0]["vc_url"], sim["vc_url"], 1.0 - sim["similarity"]))

    G = build_relationship_graph(co_investments, competitors)
    results["relationships"] = G

    # Gap analysis
    results["gap"] = analyze_gap(vc_embeddings, founder_embedding, cluster_meta)

    return results

def run_chat_agent(context, query):
    return answer_question(query, context)
