"""Stock Research Multi-Agent System

Usage:
  python main.py <股票代码或公司名> [options]

Options:
  --check         仅测试 LLM 连接，不运行分析

Examples:
  python main.py AAPL              # 苹果 (US)
  python main.py 茅台              # 贵州茅台 (A股)
  python main.py 600519.SS         # 贵州茅台 (A股)
  python main.py 腾讯              # 腾讯控股 (港股)
  python main.py Tesla             # 特斯拉 (US)
  python main.py --check           # 连接诊断
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Parse flags before imports
args = sys.argv[1:]
flags = {"--check"}
positional = [a for a in args if a not in flags]
check_only = "--check" in args

if "--help" in args or "-h" in args:
    print(__doc__)
    sys.exit(0)

from orchestrator import ResearchOrchestrator
from config import API_KEY, BASE_URL, MODEL_FAST, DEFAULT_TICKER, LLM_PROVIDER, check_llm_connection
from tools.ticker_resolver import TickerResolver


def _create_client():
    if LLM_PROVIDER == "openai":
        import openai
        return openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    else:
        import anthropic
        return anthropic.Anthropic(api_key=API_KEY, base_url=BASE_URL)


def _resolve_ticker(raw_input: str):
    """Resolve user input to a ticker. Returns (ticker, company_name)."""
    is_ticker = (
        raw_input.isupper() and len(raw_input) <= 5 and raw_input.isalpha()
        or (raw_input.isascii() and raw_input.isalpha()
            and raw_input.islower() and len(raw_input) <= 4)
        or (len(raw_input.split(".")) == 2 and raw_input.split(".")[1].isalpha())
    )

    if is_ticker:
        return raw_input.upper(), raw_input.upper()

    client = _create_client()
    resolver = TickerResolver(client, MODEL_FAST)
    result = resolver.resolve(raw_input)
    ticker = result.get("full_ticker", raw_input.upper())
    company_name = result.get("company_name", raw_input)
    print(f"  Resolved -> {ticker} ({company_name})")
    print(f"  Confidence: {result.get('confidence', '?')}")
    return ticker, company_name


def main():
    raw_input = positional[0] if positional else None

    print("===============================================")
    print("  Stock Research Multi-Agent System")
    print("===============================================")

    if check_only:
        print("\n  [check] 测试 LLM 连接...")
        ok, msg = check_llm_connection(timeout=10.0)
        print(f"  {'[OK]' if ok else '[FAIL]'} {msg}")
        return

    # No input -> interactive mode
    if not raw_input:
        print("\n  Enter a stock ticker or company name to analyze.")
        print("  Examples: AAPL, 茅台, 600519.SS, Tesla, 腾讯, BABA")
        raw_input = input("\n  > ").strip()
        if not raw_input:
            raw_input = DEFAULT_TICKER

    print(f"\n  Input: {raw_input}")
    print("  Resolving ticker...")

    ticker, company_name = _resolve_ticker(raw_input)

    print("===============================================\n")

    try:
        orchestrator = ResearchOrchestrator()
        report = orchestrator.run(ticker, company_hint=company_name)
        if report:
            # Report is saved to output/ directory; only print a summary line
            print(f"  Report generated successfully.")
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
