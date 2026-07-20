"""Agent 1: Financial Analyst — financial analysis with multi-source data."""
from utils import chat, format_estimates, latest_actual_period, today_str
from tools.data_sources import fetch_all


SYSTEM_PROMPT = """你是一名拥有10年卖方研究经验的资深财务分析师，CPA持证，擅长通过财务报表还原企业真实经营质量。

## 你的任务
基于提供的原始财务数据，输出一份结构化的财务分析报告。

## 关键约束——比什么都重要
你拿到的数据可能不完整。报告中必须严格区分「有数据支撑的结论」和「无法判断的部分」：

1. 数据有的指标 → 分析并给出结论
2. 数据没有的指标 → 必须写「数据暂不可用」，严禁用你的训练知识去估算
3. 严禁说"根据行业平均水平""通常来说""合理推测"等绕过数据缺失的措辞
4. 仅凭 PE + 市值 + 股价 能做的分析有限，诚实承认这一点比硬凑分析更有价值

## 报告结构（严格按此顺序输出）

### 零、数据可用性声明（必须放在最前面，50字以内）
明确说明本次分析有哪些数据、缺哪些数据。例如：
"本次仅有行情报价数据（股价/市值/PE/52周区间），无财务报表数据（营收/净利润/现金流/资产负债表均缺失）。以下分析基于可用数据展开，缺失部分已标注。"

### 一、公司基本面快照
- 从可用数据中提取的基本信息（名称、市值、行业）
- 如果有 business description 则引用
- 如果行业/业务描述缺失，明确标注

### 二、可得数据分析
基于实际提供的数据进行分析：
- 估值水平：当前 PE 相对 52 周区间的位置
- 市值规模：大盘/中盘/小盘，与同行业相对位置
- 如果无行业对标数据，仅描述绝对值，不做横比

### 三、缺失数据清单
逐项列出本次无法覆盖的分析维度：
- 利润表：数据不可用
- 资产负债表：数据不可用
- 现金流量表：数据不可用
- 具体哪些比率不可用

### 四、基于可得数据的核心发现（3条以内）
仅基于实际数据得出，每条标注用了哪个数据点。

## 硬性规则
1. 中文输出，Markdown 格式，800-1200 字
2. 每段分析标注数据来源
3. **宁缺毋滥**：数据不够就说不够，不要凑
4. **时间锚定**：以数据中给定的「当前日期」和「最新已披露实际期」为准。你的训练知识里的时间线是过期的——已披露实际期之前的年份（含2024/2025）一律按历史事实写，严禁写成「预测」「预计」；只有该期之后的年份才允许预测

## 输出末尾必须包含结构化摘要块（供下游Agent解析）
```json
{
  "revenue_growth": "数值%或N/A",
  "net_margin": "数值%或N/A",
  "roe": "数值%或N/A",
  "debt_to_equity": "数值或N/A",
  "fcf_trend": "上升/下降/平稳/未知",
  "financial_health_score": "1-10分或N/A",
  "key_risks": ["风险1"],
  "key_strengths": ["亮点1"],
  "data_completeness": "full/partial/quote_only"
}
```"""


class FinancialAnalystAgent:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def analyze(self, ticker: str, data: dict = None, company_hint: str = "") -> str:
        if data is None:
            data = fetch_all(ticker, llm_client=self.client, llm_model=self.model,
                           company_hint=company_hint)

        data_str = _format_data(data)
        return chat(
            self.client,
            model=self.model,
            max_tokens=4096,
            temperature=0.2,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"请分析以下数据并输出财务分析报告：\n\n{data_str}"}],
        )


def _format_data(data: dict) -> str:
    ds = data.get("data_source", "unknown")
    tried = data.get("_sources_tried", [])

    source_map = {
        "yfinance": "yfinance 实时数据（含财务报表）",
        "yahoo_v10": "Yahoo Finance v10（含财务比率）",
        "eastmoney": "东方财富 API 实时行情",
        "tencent": "腾讯行情 API 实时数据（仅行情报价，无财务报表）",
        "tencent+em_f10": "腾讯行情 + 东方财富 F10（含财报）",
        "sina": "新浪财经 API 实时行情（仅行情报价）",
        "llm_knowledge": "LLM 训练知识（非实时）",
        "none": "全部数据源均不可用",
    }

    # Detect data completeness
    has_financials = bool(data.get("financials"))
    has_ratios = bool(data.get("ratios"))
    has_full = has_financials or len(data.get("ratios", {})) > 3

    lines = ["# 原始财务数据\n"]
    lines.append(f"数据来源: {source_map.get(ds, ds)}")
    if tried:
        lines.append(f"尝试链路: {' -> '.join(tried)}")

    # === Time anchor: LLM training knowledge is stale — force current timeline ===
    lines.append(f"\n## ═══════════════════════════════════════")
    lines.append(f"## 当前日期: {today_str()}")
    lap = latest_actual_period(data)
    if lap:
        lines.append(f"## 最新已披露实际期: {lap}")
        lines.append(f"## 该期及之前的年份（含2024/2025）均为已发生的历史事实，严禁标注为「预测」；预测只能针对该期之后的财年")
    lines.append(f"## ═══════════════════════════════════════")

    # === PROMINENT data completeness section ===
    lines.append(f"\n## ═══════════════════════════════════════")
    lines.append(f"## 数据完整度: {'完整' if has_full else '仅行情报价 — 无利润表/资产负债表/现金流量表'}")
    if not has_full:
        lines.append(f"## 以下指标均不可用: 营收/净利润/ROE/毛利率/资产负债率/D/E/FCF/现金流")
        lines.append(f"## 可用指标: 股价/市值/PE/52周区间")
    lines.append(f"## ═══════════════════════════════════════")

    if data.get("error"):
        lines.append(f"\n[ERROR] {data['error']}")

    lines.append(f"\n## 基本信息")
    lines.append(f"股票: {data.get('ticker', '?')}")
    lines.append(f"名称: {data.get('name', '?')}")
    lines.append(f"行业: {data.get('sector', '?')} / {data.get('industry', '?')}")
    lines.append(f"市值: {_fmt(data, 'market_cap')}")
    lines.append(f"股价: {_fmt(data, 'current_price')} {data.get('currency', '?')}")
    lines.append(f"交易所: {data.get('exchange', '?')} | 国家: {data.get('country', '?')}")

    desc = data.get("description", "")
    if desc:
        lines.append(f"\n## 业务简介\n{desc[:600]}")
    else:
        lines.append(f"\n## 业务简介\n数据不可用")

    for label, key in [("营收", "total_revenue"), ("现金", "total_cash"),
                        ("总负债", "total_debt"), ("自由现金流", "free_cashflow"),
                        ("经营现金流", "operating_cashflow")]:
        val = data.get(key)
        if val is not None:
            lines.append(f"- {label}: {_fmt_num(val)}")

    ratios = data.get("ratios", {})
    if ratios:
        lines.append(f"\n## 关键财务比率（最新一期，来源: {ds}）")
        ratio_labels = {
            "pe_ratio": "PE", "forward_pe": "Forward PE",
            "pb_ratio": "PB", "ps_ratio": "PS",
            "debt_to_equity": "D/E", "current_ratio": "流动比率",
            "roe_pct": "ROE (%)", "profit_margin_pct": "净利润率 (%)",
            "revenue_growth_pct": "收入增长率 (%)", "gross_margin_pct": "毛利率 (%)",
            "ev_to_revenue": "EV/Revenue", "ev_to_ebitda": "EV/EBITDA",
            "beta": "Beta", "dividend_yield_pct": "股息率 (%)",
        }
        for k, v in ratios.items():
            label = ratio_labels.get(k, k)
            lines.append(f"- {label}: {v}")
    elif not has_full:
        lines.append(f"\n## 关键财务比率\n以上财务比率均不可用（仅获取到 PE 估值指标）")

    # ── Analyst consensus estimates (forward years, REAL forecast data) ──
    lines.append(f"\n## 分析师一致预期（未来年份真实预测，非本系统推测）")
    lines.append(format_estimates(data.get("estimates")))

    # ── Latest quarters (highest-frequency fundamentals — 周期拐点信号) ──
    quarterly = data.get("quarterly", [])
    if quarterly:
        lines.append(f"\n## 最新季度数据（报告期累计值）")
        for q in quarterly:
            p = str(q.get("period", ""))
            qlabel = {"0331": "Q1", "0630": "H1", "0930": "Q3累计"}.get(p[4:], p[4:]) if len(p) == 8 else p
            parts = [f"{p[:4]}年{qlabel}"]
            if q.get("revenue") is not None:
                parts.append(f"营收 {_fmt_num(q['revenue'])}")
            if q.get("net_income") is not None:
                parts.append(f"净利润 {_fmt_num(q['net_income'])}")
            if q.get("revenue_yoy_pct") is not None:
                parts.append(f"营收同比 {q['revenue_yoy_pct']:+.2f}%")
            if q.get("profit_yoy_pct") is not None:
                parts.append(f"净利同比 {q['profit_yoy_pct']:+.2f}%")
            lines.append("- " + "，".join(parts))

    # ── Multi-year metrics history (from akshare) ──
    metrics_history = data.get("metrics_history", [])
    if metrics_history and len(metrics_history) >= 2:
        lines.append(f"\n## 多年财务指标趋势（共{len(metrics_history)}个财年）")
        lines.append("")
        # Table header
        header_cols = ["指标"] + [m["period"][:4] for m in metrics_history[:5]]
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "|".join(["------"] * len(header_cols)) + "|")
        # Revenue row
        rev_row = ["营收"] + [_fmt_num(m.get("revenue")) if m.get("revenue") else "N/A" for m in metrics_history[:5]]
        lines.append("| " + " | ".join(rev_row) + " |")
        # Net income row
        ni_row = ["净利润"] + [_fmt_num(m.get("net_income")) if m.get("net_income") else "N/A" for m in metrics_history[:5]]
        lines.append("| " + " | ".join(ni_row) + " |")
        # Key ratio rows
        for rk, rl in [("revenue_growth_pct", "营收增速(%)"), ("gross_margin_pct", "毛利率(%)"),
                         ("profit_margin_pct", "净利率(%)"), ("roe_pct", "ROE(%)"),
                         ("eps", "EPS")]:
            vals = []
            for m in metrics_history[:5]:
                v = m.get(rk)
                vals.append(f"{v:.2f}" if v is not None else "N/A")
            if any(v != "N/A" for v in vals):
                lines.append("| " + rl + " | " + " | ".join(vals) + " |")
        lines.append("")

    financials = data.get("financials", {})
    if financials:
        for stmt_name in ("income", "balance", "cashflow"):
            stmt = financials.get(stmt_name, {})
            if stmt:
                label = {"income": "利润表", "balance": "资产负债表", "cashflow": "现金流量表"}.get(stmt_name, stmt_name)
                lines.append(f"\n## {label}")
                for metric, vals in stmt.items():
                    vals_str = ", ".join(f"{date}: {_fmt_num(v)}" for date, v in vals.items())
                    lines.append(f"- {metric}: {vals_str}")
    else:
        lines.append(f"\n## 财务报表数据\n全部不可用 — 未获取到利润表/资产负债表/现金流量表")

    price = data.get("price", {})
    if price:
        lines.append(f"\n## 价格数据")
        for pk, pl in [("current", "当前价"), ("high_52w", "52周高"), ("low_52w", "52周低")]:
            v = price.get(pk)
            if v is not None:
                lines.append(f"- {pl}: {v}")
        for pk, pl in [("volatility_pct", "年波动率 (%)"), ("total_return_pct", "年回报 (%)")]:
            v = price.get(pk)
            if v is not None:
                lines.append(f"- {pl}: {v}")

    return "\n".join(lines)


def _fmt(data: dict, key: str) -> str:
    v = data.get(key)
    if v is None:
        return "?"
    if isinstance(v, (int, float)) and v > 1e9:
        return f"{v/1e9:.2f}B"
    if isinstance(v, (int, float)) and v > 1e6:
        return f"{v/1e6:.1f}M"
    return str(v)


def _fmt_num(v) -> str:
    if isinstance(v, (int, float)):
        if abs(v) > 1e9:
            return f"{v/1e9:.2f}B"
        if abs(v) > 1e6:
            return f"{v/1e6:.1f}M"
        return f"{v:.2f}"
    return str(v)
