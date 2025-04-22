
import os
import json
import logging
from agents.utils import (
    load_or_generate_embedding,
    save_results_to_cache,
    load_results_from_cache,
    get_vc_urls_from_file
)
from agents.website_scraper_agent import scrape_website_text
from agents.portfolio_enricher_agent import enrich_portfolio
from agents.llm_embed_gap_match_chat import generate_founder_summary, generate_vc_summary, match_founder_to_vcs, analyze_gap, generate_chat_context
from agents.categorizer_agent import categorize_vcs
from agents.relationship_agent import compute_vc_relationships
from agents.similar_company_agent import find_similar_portfolio_companies
from agents.visualization_agent import generate_visuals

logger = logging.getLogger(__name__)

def run_orchestration(founder_docs, vc_url_file="vc_urls.txt"):
    results = {}
    
    # 1. Process founder documents
    founder_summary, founder_embedding = generate_founder_summary(founder_docs)
    results["founder_summary"] = founder_summary

    # 2. Load VC URLs
    vc_urls = get_vc_urls_from_file(vc_url_file)

    vc_summaries = {}
    vc_embeddings = {}
    enriched_data = {}

    for url in vc_urls:
        try:
            # 3. Scrape and summarize
            raw_text = scrape_website_text(url)
            portfolio_info = enrich_portfolio(url)
            summary, embedding = generate_vc_summary(url, raw_text, portfolio_info)
            vc_summaries[url] = summary
            vc_embeddings[url] = embedding
            enriched_data[url] = portfolio_info
        except Exception as e:
            logger.warning(f"Skipping {url} due to error: {e}")
            continue

    results["vc_summaries"] = vc_summaries

    # 4. Categorize VCs
    clusters = categorize_vcs(vc_embeddings)
    results["clusters"] = clusters

    # 5. Compute VC relationships
    relationships = compute_vc_relationships(enriched_data)
    results["relationships"] = relationships

    # 6. Match Founder to VCs
    matches = match_founder_to_vcs(founder_embedding, vc_embeddings)
    results["matches"] = matches

    # 7. Gap Analysis
    gap = analyze_gap(vc_embeddings, founder_embedding, clusters)
    results["gap"] = gap

    # 8. Similar Companies
    similar_companies = find_similar_portfolio_companies(founder_embedding, enriched_data)
    results["similar_companies"] = similar_companies

    # 9. Visualizations
    visuals = generate_visuals(vc_embeddings, clusters, relationships)
    results["visuals"] = visuals

    # 10. Chatbot Context
    chat_context = generate_chat_context(founder_summary, vc_summaries)
    results["chat_context"] = chat_context

    return results
