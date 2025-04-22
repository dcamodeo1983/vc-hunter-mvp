import os
import logging
from agents.website_scraper_agent import scrape_vc_website
from agents.portfolio_enricher_agent import enrich_portfolio_data
from agents.llm_embed_gap_match_chat import (
    generate_founder_summary,
    generate_vc_summary,
    match_founder_to_vcs,
    analyze_gap,
    generate_chat_context,
)
from agents.categorizer_agent import categorize_vcs
from agents.relationship_agent import build_relationship_graph
from agents.similar_company_agent import find_similar_portfolio_companies
from agents.visualization_agent import generate_tsne_plot, generate_heatmap_from_themes
from agents.chat_agent import answer_question
from agents.utils import load_documents_as_text

logger = logging.getLogger(__name__)

def run_orchestration(founder_docs, vc_urls):
    results = {}

    # Load and summarize founder documents
    founder_text = load_documents_as_text(founder_docs)
    founder_summary, founder_embedding = generate_founder_summary(founder_text)
    results["founder_summary"] = founder_summary

    # Scrape and enrich VC websites
    vc_data = []
    for url in vc_urls:
        try:
            scraped_text = scrape_vc_website(url)
            enriched_portfolio = enrich_portfolio_data(url)
            summary, embedding = generate_vc_summary(url, scraped_text, enriched_portfolio)
            vc_data.append({
                "vc_url": url,
                "summary": summary,
                "embedding": embedding,
                "portfolio": enriched_portfolio
            })
        except Exception as e:
            logger.warning(f"Skipping malformed VC entry: {url} due to {e}")

    if not vc_data:
        raise ValueError("No usable VC data was retrieved.")

    results["vc_summaries"] = [vc["summary"] for vc in vc_data]

    # Match founder to VCs
    matches = match_founder_to_vcs(founder_embedding, vc_data)
    results["matches"] = matches

    # Analyze gap
    results["gap"] = analyze_gap(founder_summary, [vc["summary"] for vc in vc_data])

    # Visuals: Cluster & Heatmap
    clusters, cluster_plot = generate_tsne_plot(vc_data)
    results["clusters"] = clusters
    results["visuals"] = {"cluster_plot": cluster_plot}
    heatmap = generate_heatmap_from_themes(vc_data)
    results["visuals"]["heatmap"] = heatmap

    # Relationship Graph
    co_investments = []
    competitors = []
    for vc in vc_data:
        sim_list = find_similar_portfolio_companies(vc, vc_data)
        if not matches:
            logger.info(f"No match found for founder – skipping {vc['vc_url']} in co-investment map.")
            continue
        for sim in sim_list:
            competitors.append((vc["vc_url"], sim["vc_url"]))
            co_investments.append((vc["vc_url"], matches[0]["vc_url"], sim["similarity"]))
    relationship_plot = build_relationship_graph(co_investments, competitors)
    results["visuals"]["relationship_plot"] = relationship_plot

    # Chat context
    context = generate_chat_context(founder_summary, [vc["summary"] for vc in vc_data])
    results["chat_context"] = context

    return results
