"""Agent 2: Sentiment Analyst — analyzes market sentiment and public opinion."""
from tools.stock_data import get_all_data
from utils import chat, format_estimates, today_str


SYSTEM_PROMPT = """你是买方机构的市场情绪分析师，有6年交易台经验，擅长从市场行为和舆论中捕捉预期差。

## 你的任务
基于公司基本面信息和市场数据，分析当前市场对该公司的情绪取向和预期定价程度。注意：你是在分析**市场怎么看这家公司**，不是你自己对公司的基本面判断。

## 反编造规则
1. 不要编造具体的新闻事件、分析师报告标题或机构名称
2. 不要编造"近期有大量看涨期权交易"等具体交易数据，除非数据中明确提供了
3. 情绪判断基于价格行为（52周位置、波动率）和基本面数据推导，不要假装读过新闻
4. 如果数据中提供了「分析师一致预期」（机构评级分布、目标价区间、预测机构数），这些是真实数据，可以直接引用并标注来源
5. 如果数据中提供了「最新新闻标题」，市场叙事分析必须锚定这些带日期的真实新闻；严禁使用训练知识里的旧周期叙事（如几年前的行业景气判断）冒充当前市场共识。没有新闻数据时，必须声明叙事仅来自基本面与价格行为推断

## 报告结构（严格按此顺序输出）

### 一、情绪温度计
- 整体情绪评分：-100（极度悲观）到 +100（极度乐观）
- 置信度：高/中/低（取决于信息充分程度）
- 情绪方向：正面 / 偏正面 / 中性 / 偏负面 / 负面

### 二、市场叙事分析
当前市场主流在讲什么故事？（如：复苏逻辑、困境反转、估值修复、成长确定性等）

### 三、正面催化因素
市场关注的积极因素，按重要性排序：
1. （因素描述 + 市场反应逻辑）
2. ...

### 四、负面担忧因素
市场主要的风险定价点，按重要性排序：
1. （风险描述 + 担忧程度）
2. ...

### 五、预期差分析
- 基本面实际情况 vs 市场预期：是否存在高估或低估？
- 哪些信息可能被市场过度定价？
- 哪些风险可能被市场忽视？

### 六、情绪与基本面的匹配度判断
市场情绪是否充分反映了财务基本面？是乐观过头还是悲观过头？给出明确判断。

### 七、信息局限性声明
明确说明本次分析基于哪些信息源、缺少哪些维度的数据（如缺少实时新闻、缺少机构研报、缺少社交舆情等），坦诚标注分析的可靠程度。

## 硬性规则
1. **区分事实与判断**：哪些是已知数据，哪些是你的推断，必须分开
2. **不与财务报告矛盾**：基本面事实以财务分析师的结论为准，不要推翻
3. **避免绝对化表述**：用倾向于、大概率、市场似乎在定价等审慎措辞
4. **诚实面对信息不足**：如果缺乏真实舆情数据，明确说明基于基本面的合理推断
5. 中文输出，Markdown格式，1000字以内

## 输出末尾必须包含结构化摘要块（供下游Agent解析）
```json
{
  "sentiment_score": -100到+100的整数,
  "sentiment_confidence": "high/medium/low",
  "key_catalysts": ["催化1", "催化2"],
  "key_risks": ["风险1", "风险2"],
  "expectation_gap": "高估/合理/低估/无法判断",
  "data_limitations": ["缺失的数据源1", "缺失的数据源2"]
}
```"""


def _format_news(news: list) -> str:
    """Render dated news headlines for the sentiment prompt."""
    if not news:
        return "数据不可用 — 禁止使用训练知识中的旧新闻/旧叙事"
    return "\n".join(
        f"- [{n.get('datetime', '?')}] {n.get('title', '')}" +
        (f"（{n['source']}）" if n.get("source") else "")
        for n in news
    )


class SentimentAnalystAgent:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def analyze(self, ticker: str, financial_report: str, data: dict = None) -> str:
        if data is None:
            data = get_all_data(ticker)
        company_name = data.get("name", ticker)
        price = data.get("price", {})
        ratios = data.get("ratios", {})

        prompt = f"""请基于以下信息进行市场情绪分析。

## 公司
{company_name} ({ticker})
当前日期: {today_str()}（你的训练知识时间线已过期，一切以该日期和数据为准）
行业: {data.get('sector', '?')}
市值: {data.get('market_cap', '?')}

## 市场数据
当前股价: {data.get('current_price', price.get('current', '?'))}
52周高: {price.get('high_52w', '?')}
52周低: {price.get('low_52w', '?')}
PE: {ratios.get('pe_ratio', '?')}
年波动率: {price.get('volatility_pct', '?')}%
年回报: {price.get('total_return_pct', '?')}%

## 分析师一致预期（真实数据，如有）
{format_estimates(data.get('estimates'))}

## 最新新闻标题（真实数据，按时间倒序）
{_format_news(data.get('news'))}

## 财务分析师的报告
以下是财务分析师对公司基本面的分析结论，请在其基础上做情绪判断：

{financial_report}

---
请输出你的情绪分析报告。"""
        response = chat(
            self.client,
            model=self.model,
            max_tokens=3072,
            temperature=0.3,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response
