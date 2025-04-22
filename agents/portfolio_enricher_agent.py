
import random

# Simulated enrichment, ideally would scrape or load known portfolio companies
def enrich_portfolio(vc_url):
    # Placeholder examples for enrichment
    dummy_data = {
        "https://a16z.com": ["Flexport", "Instacart", "OpenAI"],
        "https://foundersfund.com": ["Palantir", "Anduril", "SpaceX"],
        "https://8vc.com": ["Honor", "BuildZoom", "Mixpanel"],
        "https://luxcapital.com": ["Mythic", "Varda", "Planetary Resources"]
    }
    return dummy_data.get(vc_url, random.sample(["CompanyA", "CompanyB", "CompanyC", "CompanyD"], 3))
