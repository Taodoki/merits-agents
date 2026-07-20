# Stock Research Multi-Agent System

输入股票代码或名称，4 位 AI 分析师自动协作，输出一份完整的投资研究报告。

## 效果展示

**输入** `python main.py 茅台` 后系统自动产出：

```
output/
├── 600519_SS_20260719_105057_01_financial_analysis.md   # 财务分析师
├── 600519_SS_20260719_105057_02_sentiment_analysis.md   # 情绪分析师
├── 600519_SS_20260719_105057_03_valuation_analysis.md   # 估值分析师
└── 600519_SS_20260719_105057_FINAL_REPORT.md            # 最终投资研究报告
```

最终报告包含：投资要点（含目标价与评级）、公司业务分析、财务与盈利质量分析、市场情绪与预期差、估值（DCF + 可比公司 + PE）、投资结论、风险提示、附录财务指标。

## Agent 协作流程

```
用户输入（股票代码/公司名）
       │
       ▼
  [Ticker解析]  ──  模糊输入 → LLM解析 → 精确代码（如 茅台 → 600519.SS）
       │
       ▼
  [数据获取]    ──  多源自动切换：akshare → 东财 → 腾讯 → 新浪 → yfinance
       │
       ├──▶ Agent 1: 财务分析师（CPA）── 财报解读、财务比率、趋势分析
       │         │
       ├──▶ Agent 2: 情绪分析师 ── 市场情绪、叙事分析、预期差判断
       │         │
       ├──▶ Agent 3: 估值分析师（CFA）── DCF、可比公司、PE 估值
       │         │
       └──▶ Agent 4: 研究主管/主编 ── 整合三份报告 → 最终投资研究报告
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
cp .env.example .env
```

```env
# 选择 provider: anthropic（默认）或 openai
LLM_PROVIDER=anthropic

# Anthropic SDK
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_BASE_URL=https://api.anthropic.com

# 或 OpenAI 兼容 API（DeepSeek / Qwen / 等）
# OPENAI_API_KEY=sk-your-key
# OPENAI_BASE_URL=https://api.openai.com/v1
```

### 3. 运行

```bash
# 股票代码
python main.py AAPL
python main.py 600519.SS

# 中文名称（自动解析）
python main.py 茅台
python main.py 腾讯
python main.py Tesla

# 测试连接
python main.py --check
```

## 功能特性

### 多市场覆盖
- **A 股**（上海 .SS / 深圳 .SZ）
- **港股**（.HK）
- **美股**（无后缀）
- 支持中文名、英文名、代码片段等多种输入形式，由 LLM 自动解析

### 多数据源 + 自动容灾
数据获取链条自动 fallback，并对每个数据点标注来源：

| 数据 | A 股 | 港股 | 美股 |
|------|------|------|------|
| 行情/财报 | akshare（东财）| akshare（东财）| yfinance |
| 最新季度 | akshare 财务摘要 | akshare 财务摘要 | yfinance |
| 一致预期 | 东财 F10 | etnet | yfinance |
| 最新新闻 | 东财个股新闻 | 东财个股新闻 | 东财个股新闻 |
| 估值锚定 | 东财同行估值 + PE 分位 | 东财行业排名 | 不支持 |

### 反幻觉设计
- 数据缺失时标注"数据暂不可用"，严禁 LLM 用训练知识编造
- 每条分析标注用了哪个数据源
- 无现金流数据时 DCF 直接跳过，不捏造
- 无可比公司数据时可比法直接跳过

### 双 Provider 支持
- 支持 Anthropic SDK（Claude）和 OpenAI SDK（兼容 DeepSeek/Qwen 等）
- 通过 `.env` 中的 `LLM_PROVIDER` 一行切换

### 结构化输出
每个 Agent 输出末尾包含 JSON 摘要块（财务评分、情绪评分、估值区间等），供下游程序解析。

## 项目结构

```
stock_research_agent/
├── main.py                  # 入口，参数解析，ticker 解析
├── orchestrator.py          # 协调器，4 Agent 顺序调度
├── config.py                # Provider 配置，SDK 切换，连接检测
├── utils.py                 # LLM 调用封装，重试，数据格式化
├── agents/
│   ├── financial_analyst.py # Agent 1: 财务分析师
│   ├── sentiment_analyst.py # Agent 2: 情绪分析师
│   ├── valuation_analyst.py # Agent 3: 估值分析师
│   └── report_editor.py     # Agent 4: 研究主编
├── tools/
│   ├── data_sources.py      # 多源数据获取器（核心，1700+ 行）
│   ├── ticker_resolver.py   # 模糊输入 → 精确代码
│   ├── news_tools.py        # 新闻获取
│   └── stock_data.py        # 数据获取统一入口
├── output/                  # 分析报告输出目录
│   └── *_FINAL_REPORT.md    # 最终投资研究报告
├── .env.example             # 环境变量模板
└── requirements.txt
```

## 技术栈

- **Python 3.14+**
- **LLM SDK**: anthropic / openai
- **数据源**: akshare, yfinance
- **HTTP**: httpx, lxml

## 设计理念

1. **数据诚实优先**：不知道就说不知道，不用 AI 幻觉填补空白
2. **约束先行**：先定规则再写代码，CLAUDE.md + AGENTS.md 双重约束
3. **输出即成品**：最终报告直接可读，不需要二次加工
4. **多源备份**：单一数据源挂了分析继续跑，只在报告中标注数据缺失

## License

MIT
