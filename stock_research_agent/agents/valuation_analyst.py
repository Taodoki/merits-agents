"""Agent 3: Valuation Analyst — DCF and comparable valuation."""
from tools.stock_data import get_all_data
from utils import chat, format_estimates, latest_actual_period, today_str


SYSTEM_PROMPT = """你是CFA持证人，曾在PE基金负责投后估值，现任卖方研究所估值分析师，精通DCF、可比公司、precedent transaction等多种估值方法。

## 你的任务
基于财务数据和市场情绪，对公司进行严谨的估值分析。所有假设必须明确列出，可追溯、可质疑。

## 关键约束——数据诚实比模型完整性更重要
1. 先盘点有哪些可用数据，再决定用哪些估值方法
2. 没有FCF/经营现金流数据 → DCF模型不可行，必须跳过，不要编造FCF基数
3. 没有可比公司数据 → 可比公司法不可行，必须跳过，不要编造同行PE
4. 只有PE+EPS → 只能用PE估值法，基于当前PE和合理PE区间给出目标价
5. 所有假设参数（无风险利率、ERP、增长率等）必须明确标注来源或取值理由
6. 严禁说"根据行业平均水平"——如果不知道，就说不知道
7. **未来年份预测分级**：数据中提供「分析师一致预期」时，未来EPS/盈利必须优先采用该真实预测数据并标注来源（含机构数）——它是市场真实预测，不是你的假设；一致预期缺失时，才允许自行假设，且必须写明"无一致预期数据，以下为本模型假设"
8. **行业锚定**：数据中提供「行业与可比公司锚定」时，可比公司法必须使用其中的真实同行公司（禁止自行更换或编造对比标的），合理估值区间必须参考行业中值/行业平均与自身历史估值分位；缺少该数据时，明确说明估值无行业锚点
9. **时间锚定**：以数据中给定的「当前日期」和「最新已披露实际期」为准——你的训练知识时间线是过期的。已披露实际期及之前的年份（含2024/2025）均为历史事实，严禁当作预测基期；预测只能针对最新实际期之后的未来两个财年

## 报告结构（按数据可用性灵活输出，不可用的章节直接标注跳过）

### 一、数据可用性评估
列出本次有哪些估值相关数据、缺哪些。例如：
"可用：股价/市值/PE/EPS/ROE。缺失：FCF/经营现金流（DCF不可行）、可比公司PE（可比法不可行）。本次仅采用PE估值法。"

### 二、估值方法选择
基于数据可用性选择方法，说明每种方法是否适用及理由。

### 三、PE估值法（核心方法，数据门槛最低）
- 当前PE与52周区间位置
- 合理PE中枢判断依据（自身历史区间、盈利增速、ROE水平）
- 预测EPS：优先采用分析师一致预期（如有），缺失时才自行假设并标注
- 目标价区间 = 预测EPS × PE区间
- 如有一致预期的机构目标价区间，将其作为市场锚点进行对比（说明一致/偏离），但不盲从

### 四、DCF模型（仅在FCF数据可用时输出）
如FCF数据缺失，此章仅写一句话："FCF数据暂不可用，DCF估值无法进行。"
如数据可用，按标准DCF框架输出，含敏感性分析表。

### 五、可比公司估值（有真实同行数据时必须输出）
如无可比数据，此章仅写一句话："可比公司估值数据暂不可用，无法进行横向对比。"
如有「行业与可比公司锚定」数据，必须基于其中的真实同行输出对比表（本公司 vs 同行 vs 行业中值），并说明本公司相对行业溢价/折价的理由；同行名单以数据为准，不得替换。

### 六、估值综合结论
- 目标价格区间
- 中枢目标价
- 当前股价
- 上涨/下跌空间
- 估值评级（标注评级所依赖的假设强度：强/中/弱）

### 七、估值风险提示
哪些假设变动会显著影响估值结论？最大的不确定性来源是什么？

## 硬性规则
1. **假设透明**：每一个参数都要有取值依据，不能凭空出现
2. **区间思维**：永远给估值区间，不给单点目标价
3. **数据诚实**：如果关键数据缺失，明确标注该参数为合理假设，非实际数据
4. **宁缺毋滥**：做不了的估值方法直接跳过，不要凑
5. 中文输出，Markdown格式，1200字以内

## 输出末尾必须包含结构化摘要块（供下游Agent解析）
```json
{
  "dcf_value_per_share": "数值或N/A",
  "relative_value_per_share": "数值或N/A",
  "target_price_low": "数值",
  "target_price_high": "数值",
  "target_price_mid": "数值",
  "current_price": "数值",
  "upside_pct": "百分比数值",
  "valuation_rating": "强烈推荐/推荐/持有/回避",
  "valuation_method_used": "PE/DCF/可比公司（列出实际使用的）",
  "key_valuation_risks": ["风险1", "风险2"]
}
```"""


def _format_anchor(anchor: dict) -> str:
    """Render industry/peer valuation anchors for the valuation prompt."""
    if not anchor:
        return "数据不可用 — 估值缺少行业锚点，需明确标注"
    lines = []
    if anchor.get("pe_ttm_current") is not None:
        pct = anchor.get("pe_ttm_percentile_10y")
        lines.append(f"- 自身 PE-TTM: {anchor['pe_ttm_current']}" +
                     (f"（近10年分位 {pct}%）" if pct is not None else ""))
    for key, label in [("pe_ttm_industry_rank", "PE-TTM 行业内排名"),
                       ("pb_industry_rank", "PB 行业内排名")]:
        if anchor.get(key) is not None:
            lines.append(f"- {label}: 第 {anchor[key]} 位")
    for key, label in [("self", "本公司（行业比较口径）"),
                       ("industry_median", "行业中值"), ("industry_average", "行业平均")]:
        ind = anchor.get(key) or {}
        if ind:
            parts = [f"{k} {v}" for k, v in ind.items() if k not in ("代码", "简称")]
            lines.append(f"- {label}: " + "，".join(parts))
    peers = anchor.get("peers") or []
    if peers:
        lines.append("- 同行业可比公司（真实名单，来自东方财富行业分类，不得替换）:")
        for p in peers:
            parts = [f"{k} {v}" for k, v in p.items() if k not in ("代码",)]
            lines.append("  - " + "，".join(parts))
    return "\n".join(lines) if lines else "数据不可用 — 估值缺少行业锚点，需明确标注"


class ValuationAnalystAgent:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def analyze(self, ticker: str, financial_report: str, sentiment_report: str,
                data: dict = None) -> str:
        if data is None:
            data = get_all_data(ticker)
        company_name = data.get("name", ticker)
        price = data.get("price", {})
        ratios = data.get("ratios", {})
        financials = data.get("financials", {})

        data_str = f"""## 公司
{company_name} ({ticker})
当前日期: {today_str()}
最新已披露实际期: {latest_actual_period(data) or '未知'}
当前股价: {data.get('current_price', price.get('current', '?'))}
市值: {data.get('market_cap', '?')}
Beta: {ratios.get('beta', '?')}

## 估值相关数据
PE: {ratios.get('pe_ratio', '?')}
PB: {ratios.get('pb_ratio', '?')}
PS: {ratios.get('ps_ratio', '?')}

## 价格数据
{price}

## 现金流数据
{financials.get('cashflow', {})}

## 分析师一致预期（未来年份真实预测数据）
{format_estimates(data.get('estimates'))}

## 行业与可比公司锚定（真实数据）
{_format_anchor(data.get('valuation_anchor'))}

## 财务分析摘要
{financial_report}

## 情绪分析摘要
{sentiment_report}
"""
        response = chat(
            self.client,
            model=self.model,
            max_tokens=4096,
            temperature=0.2,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"请对{company_name}({ticker})进行估值分析：\n\n{data_str}"}],
        )
        return response
