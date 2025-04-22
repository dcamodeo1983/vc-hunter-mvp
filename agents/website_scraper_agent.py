
import requests
from bs4 import BeautifulSoup

def scrape_website_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts and styles
        for script in soup(["script", "style", "noscript"]):
            script.decompose()

        # Get visible text
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        filtered_lines = [line for line in lines if line and not line.lower().startswith("copyright")]
        return "\n".join(filtered_lines[:200])  # limit to first 200 lines

    except Exception as e:
        raise RuntimeError(f"Failed to scrape {url}: {str(e)}")
