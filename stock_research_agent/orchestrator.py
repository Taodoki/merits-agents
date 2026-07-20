"""Agent orchestration — runs the 4-agent pipeline sequentially."""

from datetime import datetime
from pathlib import Path

from config import API_KEY, BASE_URL, MODEL_FAST, MODEL_BEST, LLM_PROVIDER, check_llm_connection
from agents.financial_analyst import FinancialAnalystAgent
from agents.sentiment_analyst import SentimentAnalystAgent
from agents.valuation_analyst import ValuationAnalystAgent
from agents.report_editor import ReportEditorAgent
from tools.data_sources import fetch_all


def _create_client():
    """Create an LLM client based on configured provider."""
    if LLM_PROVIDER == "openai":
        import openai
        return openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    else:
        import anthropic
        return anthropic.Anthropic(api_key=API_KEY, base_url=BASE_URL)


class ResearchOrchestrator:
    """Orchestrates the multi-agent stock research pipeline."""

    def __init__(self):
        if not API_KEY:
            raise ValueError(
                "API key not configured.\n"
                "Create a .env file in the project root with:\n"
                "  ANTHROPIC_API_KEY=your-key-here    (for Anthropic SDK)\n"
                "  OPENAI_API_KEY=your-key-here       (for OpenAI SDK)\n"
                "  LLM_PROVIDER=anthropic|openai"
            )
        self.client = _create_client()
        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(exist_ok=True)

        ok, msg = check_llm_connection()
        if ok:
            print(f"  [conn] {msg}")
        else:
            print(f"\n  [WARN] LLM 连接失败: {msg}\n")

    def run(self, ticker: str, company_hint: str = "") -> str:
        """Run the full research pipeline for a given ticker."""
        print(f"\n{'='*60}")
        display = company_hint or ticker
        print(f"  [Stock Research] {display} ({ticker})")
        print(f"{'='*60}\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_ticker = ticker.replace(".", "_")

        # Fetch market data — real-time sources only, no LLM fallback
        print("  [data] Fetching market data...")
        data = fetch_all(ticker, llm_client=self.client, llm_model=MODEL_FAST,
                         company_hint=company_hint, allow_llm_fallback=False)
        print(f"  [data] source: {data.get('data_source', '?')}"
              f" | quality: {data.get('data_quality', '?')}"
              f" | chain: {' -> '.join(data.get('_sources_tried', []))}")

        if data.get("data_quality") in ("all_failed", "failed"):
            print("\n  [X] 无法获取实时数据，分析终止。")
            print(f"  错误: {data.get('error', 'unknown')}")
            print("  提示: 请检查网络连接，确认外部数据源可访问。")
            return ""

        # Check data completeness
        has_financials = bool(data.get("financials")) or len(data.get("ratios", {})) > 2
        if not has_financials:
            print("\n  [!!!] 注意: 仅获取到行情报价（无完整财务报表——营收/净利润/现金流等均缺失）")
            print("  分析报告将明确标注数据缺失，不会用训练知识填充。\n")

        # Agent 1: Financial Analysis
        print("  [1/4] Financial Analyst working...")
        fa = FinancialAnalystAgent(self.client, MODEL_FAST)
        financial_report = fa.analyze(ticker, data=data, company_hint=company_hint)
        self._save_intermediate(safe_ticker, timestamp, "01_financial_analysis", financial_report)
        print("  [OK] Financial analysis complete")

        # Agent 2: Sentiment Analysis
        print("  [2/4] Sentiment Analyst working...")
        sa = SentimentAnalystAgent(self.client, MODEL_FAST)
        sentiment_report = sa.analyze(ticker, financial_report, data=data)
        self._save_intermediate(safe_ticker, timestamp, "02_sentiment_analysis", sentiment_report)
        print("  [OK] Sentiment analysis complete")

        # Agent 3: Valuation
        print("  [3/4] Valuation Analyst working...")
        va = ValuationAnalystAgent(self.client, MODEL_BEST)
        valuation_report = va.analyze(ticker, financial_report, sentiment_report, data=data)
        self._save_intermediate(safe_ticker, timestamp, "03_valuation_analysis", valuation_report)
        print("  [OK] Valuation analysis complete")

        # Agent 4: Report Compilation
        print("  [4/4] Editor compiling final report...")
        re_agent = ReportEditorAgent(self.client, MODEL_BEST)
        final_report = re_agent.compile(ticker, financial_report, sentiment_report,
                                        valuation_report, data=data)
        self._save_final(safe_ticker, timestamp, final_report)
        print("  [OK] Final report complete\n")

        return final_report

    def _save_intermediate(self, ticker: str, timestamp: str, name: str, content: str):
        path = self.output_dir / f"{ticker}_{timestamp}_{name}.md"
        path.write_text(content, encoding="utf-8")
        print(f"       -> saved: {path.name}")

    def _save_final(self, ticker: str, timestamp: str, content: str):
        path = self.output_dir / f"{ticker}_{timestamp}_FINAL_REPORT.md"
        path.write_text(content, encoding="utf-8")
        print(f"\n  *** Final Report: {path.name}")
        print(f"  *** {path.resolve()}")
