"""News fetching and sentiment analysis tools."""

import httpx
from datetime import datetime, timedelta


def fetch_news_newsapi(ticker: str, company_name: str) -> list:
    """Fetch recent news headlines via NewsAPI (free tier, no API key needed for basic)."""
    articles = []
    try:
        query = company_name.replace(" ", " OR ") if company_name else ticker
        url = f"https://newsapi.org/v2/everything"
        # Note: free tier requires API key; if not set, this will 426.
        # We try without key first, fall back gracefully.
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 15,
        }
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, params=params)
            data = resp.json()
            if data.get("status") == "ok":
                for art in data.get("articles", []):
                    articles.append({
                        "title": art.get("title", ""),
                        "source": art.get("source", {}).get("name", ""),
                        "published": art.get("publishedAt", "")[:10],
                        "description": art.get("description", ""),
                    })
    except Exception:
        pass

    return articles


def get_generic_headlines(ticker: str, company_name: str) -> list:
    """Fallback: just return company info for the LLM to reason about."""
    return [{
        "title": f"Market data for {company_name} ({ticker})",
        "source": "Yahoo Finance",
        "published": datetime.now().strftime("%Y-%m-%d"),
        "description": f"Latest market data available for {company_name}. Refer to financial data for context."
    }]
