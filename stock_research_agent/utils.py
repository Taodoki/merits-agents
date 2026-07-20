"""Shared LLM helpers — supports both Anthropic SDK and OpenAI SDK.

Uses the provider configured in config.LLM_PROVIDER.
"""

import time
from datetime import datetime

from config import LLM_PROVIDER


def today_str() -> str:
    """Current date for anchoring agent prompts (LLM training knowledge is stale)."""
    return datetime.now().strftime("%Y-%m-%d")


def latest_actual_period(data: dict) -> str:
    """Compute the latest ALREADY-REPORTED period in the data dict.

    Anything at or before this period is historical fact, not a forecast.
    Returns "" when unknown.
    """
    periods = []
    mh = data.get("metrics_history") or []
    if mh:
        periods.append(str(mh[0].get("period", "")))
    for stmt in (data.get("financials") or {}).values():
        for vals in stmt.values():
            periods.extend(str(k) for k in vals.keys())
    q = data.get("quarterly") or []
    if q:
        periods.append(str(q[0].get("period", "")))
    est = data.get("estimates") or {}
    for y in est.get("years", []):
        if y.get("mark") == "A":
            periods.append(str(y.get("year")))
    periods = [p for p in periods if p]
    return max(periods) if periods else ""


def format_estimates(est: dict) -> str:
    """Render analyst consensus estimates (盈利预测) for agent prompts."""
    if not est or not est.get("years"):
        return "数据不可用 — 未来年份数字均为模型假设，必须明确标注"
    cur = est.get("currency", "")
    src_name = {"eastmoney": "东方财富一致预期", "ths": "同花顺一致预期",
                "etnet": "etnet 券商一致预期", "yfinance": "Yahoo Finance 分析师预测"
                }.get(est.get("est_source"), est.get("est_source", "?"))
    cnt = est.get("analyst_count")
    lines = [f"来源: {src_name}" + (f"（{cnt} 家机构）" if cnt else "")]
    for y in est["years"]:
        mark = "实际值" if y.get("mark") == "A" else "预测"
        parts = [f"{y.get('year')}（{mark}）"]
        if y.get("eps") is not None:
            eps_str = f"EPS {y['eps']} {cur}"
            if y.get("eps_min") is not None and y.get("eps_max") is not None:
                eps_str += f"（区间 {y['eps_min']} ~ {y['eps_max']}）"
            parts.append(eps_str)
        if y.get("revenue") is not None:
            parts.append(f"营收 {y['revenue']}")
        if y.get("net_profit") is not None:
            parts.append(f"净利润 {y['net_profit']}")
        if y.get("eps_growth_pct") is not None:
            parts.append(f"EPS增速 {y['eps_growth_pct']}%")
        lines.append("- " + "，".join(parts))
    tp = est.get("target_price") or {}
    if tp:
        tp_str = f"- 机构目标价区间: {tp.get('min', '?')} ~ {tp.get('max', '?')} {cur}"
        if tp.get("median") is not None:
            tp_str += f"（中位数 {tp['median']}）"
        lines.append(tp_str)
    rt = est.get("ratings") or {}
    if rt:
        rt_map = {"buy": "买入", "add": "增持", "neutral": "中性", "reduce": "减持", "sell": "卖出"}
        lines.append("- 机构评级分布: " + "，".join(f"{rt_map.get(k, k)} {v} 家" for k, v in rt.items()))
    return "\n".join(lines)


def _chat_anthropic(client, *, model, max_tokens, temperature=0.2, system=None,
                    messages, retries=3, backoff=2.0):
    """Call Anthropic messages API with retry. Returns text content."""
    import anthropic

    kwargs = dict(model=model, max_tokens=max_tokens,
                  temperature=temperature, messages=messages)
    if system is not None:
        kwargs["system"] = system

    _RETRYABLE = (
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
    )

    last_err = None
    for attempt in range(retries + 1):
        try:
            response = client.messages.create(**kwargs)
            parts = [block.text for block in response.content if block.type == "text"]
            return "\n".join(parts)
        except _RETRYABLE as e:
            last_err = e
            if attempt == retries:
                break
            wait = backoff * (2 ** attempt)
            print(f"       [retry] LLM call failed ({type(e).__name__}), "
                  f"retrying in {wait:.0f}s ({attempt + 1}/{retries})...")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code in (500, 502, 503, 529) and attempt < retries:
                wait = backoff * (2 ** attempt)
                print(f"       [retry] API {e.status_code}, "
                      f"retrying in {wait:.0f}s ({attempt + 1}/{retries})...")
                time.sleep(wait)
                last_err = e
            else:
                raise
    raise last_err


def _chat_openai(client, *, model, max_tokens, temperature=0.2, system=None,
                 messages, retries=3, backoff=2.0):
    """Call OpenAI chat API with retry. Returns text content."""
    import openai

    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.extend(messages)

    _RETRYABLE = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )

    last_err = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=msgs,
            )
            return response.choices[0].message.content or ""
        except _RETRYABLE as e:
            last_err = e
            if attempt == retries:
                break
            wait = backoff * (2 ** attempt)
            print(f"       [retry] LLM call failed ({type(e).__name__}), "
                  f"retrying in {wait:.0f}s ({attempt + 1}/{retries})...")
            time.sleep(wait)
        except openai.APIStatusError as e:
            if e.status_code in (500, 502, 503, 529) and attempt < retries:
                wait = backoff * (2 ** attempt)
                print(f"       [retry] API {e.status_code}, "
                      f"retrying in {wait:.0f}s ({attempt + 1}/{retries})...")
                time.sleep(wait)
                last_err = e
            else:
                raise
    raise last_err


def chat(client, *, model, max_tokens, temperature=0.2, system=None,
         messages, retries=3, backoff=2.0):
    """Call the LLM with retry. Returns text content directly.

    Dispatches to the correct SDK based on LLM_PROVIDER config.
    """
    if LLM_PROVIDER == "openai":
        return _chat_openai(client, model=model, max_tokens=max_tokens,
                            temperature=temperature, system=system,
                            messages=messages, retries=retries, backoff=backoff)
    else:
        return _chat_anthropic(client, model=model, max_tokens=max_tokens,
                               temperature=temperature, system=system,
                               messages=messages, retries=retries, backoff=backoff)
