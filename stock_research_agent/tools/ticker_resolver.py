"""Fuzzy ticker resolver — uses LLM to convert company names and keywords to ticker symbols.

Examples:
  "茅台"       → "600519.SS"
  "Apple"      → "AAPL"
  "腾讯"       → "0700.HK"
  "阿里巴巴"   → "BABA"
  "特斯拉"     → "TSLA"
"""

import json
from utils import chat

RESOLVER_PROMPT = """You are a stock ticker resolver. Given any input (company name, ticker fragment, Chinese name, or keyword), identify the most likely stock ticker symbol and exchange.

Return ONLY a valid JSON object:
{
  "ticker": "THE_TICKER",
  "exchange_suffix": ".SS or .HK or .TO etc, or empty string",
  "full_ticker": "TICKER.SUFFIX or just TICKER for US stocks",
  "company_name": "Full company name in English",
  "name_cn": "公司中文名（如有）",
  "confidence": "high/medium/low",
  "note": "brief explanation of the mapping"
}

Rules:
- US stocks: ticker only, no suffix (e.g., "AAPL", "TSLA", "BABA")
- Shanghai A-shares: ticker + ".SS" suffix (e.g., "600519.SS", "601318.SS")
- Shenzhen A-shares: ticker + ".SZ" suffix (e.g., "000858.SZ", "002415.SZ")
- Hong Kong: ticker + ".HK" suffix (e.g., "0700.HK", "9988.HK")
- If the input is already a valid ticker, just return it with the correct suffix
- If unsure, guess the most likely ticker and set confidence to "medium" or "low"
- Always include a note explaining the reasoning
"""


class TickerResolver:
    """Resolves fuzzy text inputs to stock ticker symbols using LLM."""

    def __init__(self, client, model):
        self.client = client
        self.model = model

    def resolve(self, raw_input: str) -> dict:
        """Convert any input string to a structured ticker result."""
        prompt = f'Input: "{raw_input}"\n\nReturn the JSON mapping for this stock:'
        try:
            text = chat(
                self.client,
                model=self.model,
                max_tokens=1024,
                temperature=0.1,
                system=RESOLVER_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            result = json.loads(text.strip())
            # Ensure full_ticker exists
            if "full_ticker" not in result:
                ticker = result.get("ticker", raw_input)
                suffix = result.get("exchange_suffix", "")
                result["full_ticker"] = f"{ticker}{suffix}"
            return result
        except Exception as e:
            # Fallback: treat input as raw ticker
            raw = raw_input.strip().upper()
            return {
                "ticker": raw,
                "exchange_suffix": "",
                "full_ticker": raw,
                "company_name": raw,
                "name_cn": raw_input,
                "confidence": "low",
                "note": f"LLM resolution failed ({e}), using raw input as ticker"
            }


def resolve_ticker(client, model, raw_input: str) -> tuple:
    """Convenience function: returns (full_ticker, company_name)."""
    resolver = TickerResolver(client, model)
    result = resolver.resolve(raw_input)
    return result.get("full_ticker", raw_input.upper()), result.get("company_name", raw_input)
