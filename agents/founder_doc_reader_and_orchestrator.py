# agents/founder_doc_reader_and_orchestrator.py

import logging
from agents.website_scraper_agent import scrape_vc_website
from agents.portfolio_enricher_agent import enrich_portfolio_data
from agents.llm_embed_gap_match_chat import (
    generate_founder_summary,
    generate_vc_summary,
    match_founder_to_vcs,
    analyze_gap
)
from agents.utils import safe_truncate_text
from agents.relationship_agent import build_relationship_graph
from agents.visualization_agent import generate_cluster_plot
from agents.similar_company_agent import find_similar_companies

logger = logging.getLogger(__name__)

def run_full_pipeline(founder_doc_bytes, vc_urls):
    try:
        # Decode bytes assuming UTF-8 plain text or fallback
        try:
            text = founder_doc_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = founder_doc_bytes.decode("latin1")

        founder_summary, founder_embedding = generate_founder_summary(text)

        vc_summaries = []
        vc_embeddings = []

        for i, url in enumerate(vc_urls):
            raw_text, portfolio_links = scrape_vc_website(url)
            enriched = enrich_portfolio_data(portfolio_links)
            summary, embedding = generate_vc_summary(url, raw_text, enriched)
            vc_summaries.append({"url": url, "summary": summary})
            vc_embeddings.append({
                "url": url,
                "embedding": embedding,
                "portfolio": enriched,
                "summary": summary,               # Ensure summary is present
                "cluster": i,                      # Dummy cluster ID
                "theme": f"Cluster {i}"            # Dummy theme name
            })

        matches = match_founder_to_vcs(founder_embedding, vc_embeddings, vc_summaries)
        gap_insights = analyze_gap(founder_summary, [vc['summary'] for vc in vc_summaries])
        similar_companies = find_similar_companies(founder_embedding, vc_embeddings)

        cluster_plot = generate_cluster_plot(vc_embeddings)
        relationship_plot = build_relationship_graph(vc_embeddings, similar_companies)

        return {
            "founder_summary": founder_summary,
            "vc_summaries": vc_summaries,
            "matches": matches,
            "gap": gap_insights,
            "similar_companies": similar_companies,
            "visuals": {
                "clusters": cluster_plot,
                "relationships": relationship_plot
            }
        }

    except Exception as e:
        logger.error("Error in full pipeline execution", exc_info=True)
        raise e
