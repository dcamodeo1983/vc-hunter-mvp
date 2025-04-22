
import requests
from bs4 import BeautifulSoup

def scrape_vc_website(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return "", []

        soup = BeautifulSoup(response.text, "html.parser")
        text_parts = []

        for tag in soup.find_all(["p", "h1", "h2", "h3", "li"]):
            if tag.text and len(tag.text.strip()) > 40:
                text_parts.append(tag.text.strip())

        full_text = "\n".join(text_parts)[:3000]

        # Extract real portfolio links from the same domain
        base_domain = url.split("//")[-1].split("/")[0]
        portfolio_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text().lower()
            if "portfolio" in text or "investments" in text:
                if href.startswith("/"):
                    href = f"https://{base_domain}{href}"
                elif href.startswith("http"):
                    pass
                else:
                    href = f"https://{base_domain}/{href}"
                portfolio_links.append(href)

        return full_text, list(set(portfolio_links))[:10]

    except Exception as e:
        return "", []
