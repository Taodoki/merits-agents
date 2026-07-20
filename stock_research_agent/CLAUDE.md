# 个股深度研究多 Agent 系统

> 个人工作规则见 [AGENTS.md](AGENTS.md)，每次动手前先阅读。

## 项目简介
输入股票代码 → 4位 AI 分析师协作输出完整投资研究报告。

## Agent 团队
| Agent | 角色 | 职责 |
|-------|------|------|
| 财务分析师 | 资深财务分析师 | 财报解读、财务比率、趋势分析 |
| 情绪分析师 | 市场情绪分析师 | 市场情绪、舆论判断 |
| 估值分析师 | 估值分析师(CFA) | DCF估值、可比公司分析 |
| 研究主管 | 研究主管/主编 | 整合报告、最终输出 |

## 使用方式
```bash
python main.py <股票代码>
# 示例: python main.py AAPL / python main.py 600519.SS
```

## 技术栈
- Python 3.14+, anthropic SDK, yfinance
- 多 Agent 顺序协作架构（非 CrewAI，自有实现）

## 数据来源
- 行情/历史财报: akshare（东财）→ 东财 → 腾讯 → 新浪 → yfinance（按市场自动排序，见 `tools/data_sources.py`）
- 最新季度数据: akshare 财务摘要中的非年报期（`quarterly` 字段），周期拐点看这里
- 分析师一致预期（未来年份真实预测，`fetch_estimates`）:
  - A股 = 东财 F10 `RPT_WEB_RESPREDICT`（同花顺 THS 兜底），含 EPS 预测/评级分布/机构目标价
  - 港股 = etnet 综合盈利预测（akshare）
  - 美股 = yfinance analyst estimates（Yahoo 限流时不可用，报告中须标注为模型假设）
- 最新新闻（`fetch_news`）: 东财个股新闻（A股/港股/美股均支持），情绪分析的叙事必须锚定带日期的真实新闻，禁用训练知识旧叙事
- 估值锚定（`fetch_valuation_anchor`）: A股 = 东财同行估值比较（真实同行名单+行业中值/平均）+ 百度股市通10年PE分位；港股 = 东财行业排名；美股 = 不可用
- 约定：报告里未来年份的数字必须来自一致预期；可比公司必须使用数据提供的真实同行名单；缺失时明确标注
