"""Multi-source financial data fetcher.

Data source chain (market-aware ordering):
- A-shares (.SS/.SZ/.SH/.BJ): East Money -> Tencent -> Sina -> yfinance
- Hong Kong (.HK):            East Money -> Tencent -> yfinance -> Sina
- US / other:                 yfinance -> East Money -> Tencent -> Sina

LLM knowledge fallback is OFF by default — stale training data must be opted into.
"""

import json
import httpx

from utils import chat

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


# ── Source 1: yfinance ──

import time as _time

_yf_circuit_open_until = 0  # timestamp until which yfinance is skipped


def _is_rate_limit(err: Exception) -> bool:
    msg = str(err).lower()
    return ("rate limit" in msg or "ratelimit" in type(err).__name__.lower()
            or "too many requests" in msg)


def try_yfinance(ticker: str) -> tuple:
    """Fetch stock data via yfinance. Returns (data_dict, error_reason).

    Circuit breaker is time-based: opens for 60s after a rate-limit, then auto-resets.
    """
    global _yf_circuit_open_until
    now = _time.time()
    if now < _yf_circuit_open_until:
        remaining = int(_yf_circuit_open_until - now)
        return None, f"circuit_breaker: yfinance skipped for {remaining}s"
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            return None, "yfinance returned empty or missing price field"
        return _parse_yfinance(ticker, info, stock), None
    except Exception as e:
        if _is_rate_limit(e):
            _yf_circuit_open_until = now + 60
            return None, f"yfinance rate-limited (will retry after 60s)"
        return None, f"yfinance {type(e).__name__}: {e}"


def _parse_yfinance(ticker, info, stock):
    def _safe(val, fmt=None):
        if val is None:
            return None
        if fmt == "int":
            try:
                return int(val)
            except: return None
        if fmt == "pct":
            try:
                return round(float(val) * 100, 2) if val else None
            except: return None
        try:
            return round(float(val), 2) if val else None
        except: return val

    data = {"ticker": ticker, "data_quality": "yfinance", "data_source": "yfinance"}
    data["name"] = info.get("longName") or info.get("shortName") or ticker
    data["sector"] = info.get("sector")
    data["industry"] = info.get("industry")
    data["market_cap"] = _safe(info.get("marketCap"), "int")
    data["current_price"] = _safe(info.get("currentPrice") or info.get("regularMarketPrice"))
    data["currency"] = info.get("currency")
    data["exchange"] = info.get("exchange")
    data["country"] = info.get("country")
    data["employees"] = _safe(info.get("fullTimeEmployees"), "int")
    data["description"] = info.get("longBusinessSummary")

    ratios = {}
    for k, v in [
        ("pe_ratio", info.get("trailingPE") or info.get("forwardPE")),
        ("forward_pe", info.get("forwardPE")),
        ("pb_ratio", info.get("priceToBook")),
        ("ps_ratio", info.get("priceToSalesTrailing12Months")),
        ("debt_to_equity", info.get("debtToEquity")),
        ("current_ratio", info.get("currentRatio")),
        ("roe_pct", _safe(info.get("returnOnEquity"), "pct")),
        ("profit_margin_pct", _safe(info.get("profitMargins"), "pct")),
        ("revenue_growth_pct", _safe(info.get("revenueGrowth"), "pct")),
        ("beta", _safe(info.get("beta"))),
        ("dividend_yield_pct", _safe(info.get("dividendYield"), "pct")),
    ]:
        if v is not None:
            ratios[k] = v
    if ratios:
        data["ratios"] = ratios

    for stmt_name, stmt_data in [("income", stock.financials), ("balance", stock.balance_sheet), ("cashflow", stock.cashflow)]:
        if stmt_data is not None and not stmt_data.empty:
            if "financials" not in data:
                data["financials"] = {}
            extracted = _extract_financials(stmt_data, stmt_name)
            if extracted:
                data["financials"][stmt_name] = extracted

    try:
        hist = stock.history(period="1y")
        if hist is not None and not hist.empty:
            close = hist["Close"]
            data["price"] = {
                "current": _safe(float(close.iloc[-1])),
                "high_52w": _safe(float(close.max())),
                "low_52w": _safe(float(close.min())),
                "volatility_pct": _safe(close.pct_change().std() * (252 ** 0.5) * 100),
                "total_return_pct": _safe((close.iloc[-1] / close.iloc[0] - 1) * 100),
            }
    except Exception:
        pass

    return data


_FIN_STMT_METRICS = {
    "income": ["Total Revenue", "Operating Income", "Net Income", "EBITDA", "Gross Profit"],
    "balance": ["Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity",
                 "Cash And Cash Equivalents", "Total Debt", "Current Assets", "Current Liabilities"],
    "cashflow": ["Operating Cash Flow", "Free Cash Flow", "Capital Expenditure", "Dividends Paid"],
}


def _extract_financials(df, stmt_type: str):
    result = {}
    for m in _FIN_STMT_METRICS.get(stmt_type, []):
        if m in df.index:
            vals = df.loc[m].dropna().head(3)
            if not vals.empty:
                result[m] = {str(k.date()) if hasattr(k, "date") else str(k): round(float(v), 2)
                             for k, v in vals.items()}
    return result


# ── Source 2: East Money API ──

EM_SECID_MAP = {"SS": 1, "SZ": 0, "SH": 1, "HK": 116}
EM_US_PREFIX = 105


def _num(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def try_eastmoney(ticker: str) -> tuple:
    em_ticker = _to_eastmoney_format(ticker)
    if not em_ticker:
        return None, f"eastmoney cannot map ticker {ticker}"

    try:
        secid = em_ticker["secid"]
        url = "https://push2.eastmoney.com/api/qt/stock/get"
        params = {
            "secid": secid,
            "fields": "f43,f44,f45,f46,f47,f48,f50,f57,f58,f100,f116,f117,f162,f167,f168,f169,f170",
            "invt": 2, "fltt": 2,
        }
        with httpx.Client(timeout=10, headers=_HEADERS) as client:
            resp = client.get(url, params=params)
            data = resp.json()

        if data.get("data") is None:
            return None, "eastmoney returned null data"

        d = data["data"]
        price = _num(d.get("f43"))
        if not price:
            return None, "eastmoney returned no price"

        is_cn = secid.startswith(("1.", "0."))
        result = {
            "ticker": ticker,
            "data_quality": "eastmoney",
            "data_source": "eastmoney",
            "name": d.get("f58") or ticker,
            "current_price": price,
            "industry": d.get("f100"),
            "exchange": "Shanghai" if secid.startswith("1.") else "Shenzhen" if is_cn else em_ticker.get("exchange", ""),
            "currency": "CNY" if is_cn else "HKD" if secid.startswith("116.") else "USD",
        }

        market_cap = _num(d.get("f116"))
        if market_cap:
            result["market_cap"] = int(market_cap)

        ratios = {}
        for k, raw in [("pe_ratio", d.get("f162")), ("pb_ratio", d.get("f167"))]:
            v = _num(raw)
            if v is not None:
                ratios[k] = round(v, 2)

        high, low = _num(d.get("f44")), _num(d.get("f45"))
        if high or low:
            result["price"] = {"current": price, "high_52w": high, "low_52w": low}

        fin = _em_f10_financials(ticker)
        if fin:
            ratios.update(fin.pop("ratios", {}))
            result["financials"] = fin["financials"]

        if ratios:
            result["ratios"] = ratios
        return result, None
    except Exception as e:
        return None, f"eastmoney {type(e).__name__}: {e}"


def _to_eastmoney_format(ticker: str) -> dict:
    ticker = ticker.upper().strip()
    if "." not in ticker:
        return {"secid": f"{EM_US_PREFIX}.{ticker}", "exchange": "NASDAQ/NYSE"}
    parts = ticker.split(".")
    code = parts[0]
    suffix = parts[1]
    if suffix == "HK":
        return {"secid": f"116.{int(code):05d}", "exchange": "HKEX"}
    elif suffix in EM_SECID_MAP:
        return {"secid": f"{EM_SECID_MAP[suffix]}.{code}", "exchange": suffix}
    return None


def _em_secucode(ticker: str) -> str:
    t = ticker.upper()
    if "." not in t:
        return ""
    code, suffix = t.split(".")
    if suffix in ("SS", "SH"):
        return f"{code}.SH"
    if suffix == "SZ":
        return f"{code}.SZ"
    return ""


def _em_f10_financials(ticker: str) -> dict:
    secucode = _em_secucode(ticker)
    if not secucode:
        return {}
    try:
        url = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
        params = {
            "reportName": "RPT_F10_FINANCE_MAINFINADATA",
            "columns": "ALL",
            "filter": f'(SECUCODE="{secucode}")',
            "pageNumber": 1, "pageSize": 4,
            "sortTypes": -1, "sortColumns": "REPORT_DATE",
            "source": "HSF10", "client": "PC",
        }
        with httpx.Client(timeout=10, headers=_HEADERS) as client:
            resp = client.get(url, params=params)
            payload = resp.json()

        rows = (payload.get("result") or {}).get("data") or []
        if not rows:
            return {}

        revenue, net_income = {}, {}
        for row in rows:
            date = str(row.get("REPORT_DATE") or "")[:10]
            if not date:
                continue
            rev, ni = _num(row.get("TOTALOPERATEREVE")), _num(row.get("PARENTNETPROFIT"))
            if rev is not None:
                revenue[date] = round(rev, 2)
            if ni is not None:
                net_income[date] = round(ni, 2)
        if not revenue and not net_income:
            return {}

        latest = rows[0]
        ratios = {}
        for k, raw in [("roe_pct", latest.get("ROEJQ")),
                       ("gross_margin_pct", latest.get("XSMLL")),
                       ("debt_ratio_pct", latest.get("ZCFZL")),
                       ("eps", latest.get("EPSJB")),
                       ("revenue_growth_pct", latest.get("TOTALOPERATEREVETZ")),
                       ("profit_growth_pct", latest.get("PARENTNETPROFITTZ"))]:
            v = _num(raw)
            if v is not None:
                ratios[k] = round(v, 2)

        financials = {"income": {}}
        if revenue:
            financials["income"]["Total Revenue"] = revenue
        if net_income:
            financials["income"]["Net Income"] = net_income
        return {"ratios": ratios, "financials": financials}
    except Exception:
        return {}


# ── Source 3: Tencent quote API ──

def try_tencent(ticker: str) -> tuple:
    """Fetch basic quote from Tencent's quote API.

    Field positions differ by market — see comments inline.
    """
    tc_code = _to_tencent_format(ticker)
    if not tc_code:
        return None, f"tencent cannot map ticker {ticker}"
    try:
        with httpx.Client(timeout=8, headers=_HEADERS) as client:
            resp = client.get(f"https://qt.gtimg.cn/q={tc_code}")
            text = resp.content.decode("gbk", errors="replace")

        if '"' not in text:
            return None, "tencent returned malformed response"
        parts = text.split('"')[1].split("~")
        if len(parts) < 5:
            return None, "tencent returned too few fields"

        price = _num(parts[3])
        if not price:
            return None, "tencent returned no price"

        # Determine market type from Tencent code prefix
        is_us = tc_code.startswith("us")
        is_hk = tc_code.startswith("hk")
        is_cn = tc_code[:2] in ("sh", "sz")

        currency = "HKD" if is_hk else "CNY" if is_cn else "USD"

        # ── Field index map by market ──
        # All markets: name=[1], price=[3], prev_close=[4], high=[33], low=[34],
        #              PE=[39], market_cap(100M)=[44]
        # A-shares:  PB=[46], 52w_high=[47], 52w_low=[48], no English name field
        # HK:        PB at [47] (may not be reliable), 52w_high=[48], 52w_low=[49], English name=[46]
        # US:        [46]=English name, [47]=unreliable (NOT consistently PB), 52w_high=[48], 52w_low=[49]
        #
        # We only use PB from A-shares (field [46]) where it's verified.
        # For HK/US, PB is too unreliable in the Tencent quote feed — skip it.

        if is_cn:
            idx_pb, idx_52h, idx_52l = 46, 47, 48
            name_en = None
        else:
            idx_pb, idx_52h, idx_52l = None, 48, 49  # PB not used for HK/US
            name_en = parts[46] if len(parts) > 46 and parts[46] and not parts[46].startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")) else None

        name = name_en or parts[1] or ticker

        result = {
            "ticker": ticker,
            "data_quality": "tencent",
            "data_source": "tencent",
            "name": name,
            "current_price": price,
            "currency": currency,
        }

        # Price data
        high = _num(parts[idx_52h]) if len(parts) > idx_52h else None
        low = _num(parts[idx_52l]) if len(parts) > idx_52l else None
        if high or low:
            result["price"] = {"current": price}
            if high:
                result["price"]["high_52w"] = high
            if low:
                result["price"]["low_52w"] = low

        # Ratios
        ratios = {}
        pe = _num(parts[39]) if len(parts) > 39 else None
        if pe:
            ratios["pe_ratio"] = round(pe, 2)
        pb = _num(parts[idx_pb]) if idx_pb is not None and len(parts) > idx_pb else None
        if pb:
            ratios["pb_ratio"] = round(pb, 2)
        if ratios:
            result["ratios"] = ratios

        # Market cap: all markets use parts[44] in 100M units
        mcap = _num(parts[44]) if len(parts) > 44 else None
        if mcap:
            result["market_cap"] = int(mcap * 1e8)

        return result, None
    except Exception as e:
        return None, f"tencent {type(e).__name__}: {e}"


def _to_tencent_format(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if "." not in ticker:
        return f"us{ticker}"
    code, suffix = ticker.split(".")
    if suffix in ("SS", "SH"):
        return f"sh{code}"
    elif suffix == "SZ":
        return f"sz{code}"
    elif suffix == "HK":
        return f"hk{int(code):05d}"
    return None


# ── Source 4: Sina Finance API ──

def try_sina(ticker: str) -> tuple:
    sina_code = _to_sina_format(ticker)
    if not sina_code:
        return None, f"sina cannot map ticker {ticker}"
    try:
        url = f"https://hq.sinajs.cn/list={sina_code}"
        headers = {"Referer": "https://finance.sina.com.cn", **_HEADERS}
        with httpx.Client(timeout=8, headers=headers) as client:
            resp = client.get(url)
            text = resp.content.decode("gbk", errors="replace")

        if '"' not in text:
            return None, "sina returned malformed response"

        data = text.split('"')[1]
        parts = data.split(",")
        if len(parts) < 30:
            return None, "sina returned too few fields"

        name = parts[0]
        price = float(parts[3]) if parts[3] else None
        high = float(parts[4]) if parts[4] else None
        low = float(parts[5]) if parts[5] else None

        if not price:
            return None, "sina returned no price"

        result = {
            "ticker": ticker,
            "data_quality": "sina",
            "data_source": "sina",
            "name": name or ticker,
            "current_price": price,
            "currency": "CNY",
            "exchange": "China A",
        }
        if high:
            result["price"] = {"current": price, "high_52w": high, "low_52w": low}
        return result, None
    except Exception as e:
        return None, f"sina {type(e).__name__}: {e}"


def _to_sina_format(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if "." not in ticker:
        return f"gb_{ticker}"
    code, suffix = ticker.split(".")
    if suffix in ("SS", "SH"):
        return f"sh{code}"
    elif suffix == "SZ":
        return f"sz{code}"
    elif suffix == "HK":
        return f"hk{code}"
    return None


# ── Source 5: Yahoo Finance v10 direct API ──

def try_yahoo_v10(ticker: str) -> tuple:
    """Fetch financial ratios and metadata from Yahoo Finance v10 API directly.

    This bypasses the yfinance library and uses a different endpoint that
    may work even when yfinance is rate-limited.
    Returns (data_dict, error_reason).
    """
    try:
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
        params = {
            "modules": "price,summaryDetail,defaultKeyStatistics,financialData,assetProfile",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }
        with httpx.Client(timeout=12, headers=headers, follow_redirects=True) as client:
            resp = client.get(url, params=params)
            if resp.status_code != 200:
                return None, f"yahoo_v10 HTTP {resp.status_code}"
            data = resp.json()

        result_list = data.get("quoteSummary", {}).get("result", [])
        if not result_list:
            return None, "yahoo_v10 returned no result"
        r = result_list[0]

        price_data = r.get("price", {})
        price = (_raw(price_data, "regularMarketPrice") or
                 _raw(price_data, "regularMarketOpen"))
        if not price:
            return None, "yahoo_v10 returned no price"

        result = {
            "ticker": ticker,
            "data_quality": "yahoo_v10",
            "data_source": "yahoo_v10",
            "name": price_data.get("longName") or price_data.get("shortName") or ticker,
            "current_price": round(price, 2),
            "currency": price_data.get("currency"),
            "market_cap": int(_raw(price_data, "marketCap")) if _raw(price_data, "marketCap") else None,
            "exchange": price_data.get("exchangeName"),
        }

        # Summary detail (valuation ratios)
        sd = r.get("summaryDetail", {})
        ratios = {}
        _extract_yf_field(ratios, sd, "trailingPE", "pe_ratio")
        _extract_yf_field(ratios, sd, "forwardPE", "forward_pe")
        _extract_yf_field(ratios, sd, "priceToBook", "pb_ratio")
        _extract_yf_field(ratios, sd, "dividendYield", "dividend_yield_pct", is_pct=True)
        _extract_yf_field(ratios, sd, "beta", "beta")

        # Financial data (margins, growth, etc.)
        fd = r.get("financialData", {})
        _extract_yf_field(ratios, fd, "returnOnEquity", "roe_pct", is_pct=True)
        _extract_yf_field(ratios, fd, "profitMargins", "profit_margin_pct", is_pct=True)
        _extract_yf_field(ratios, fd, "revenueGrowth", "revenue_growth_pct", is_pct=True)
        _extract_yf_field(ratios, fd, "grossMargins", "gross_margin_pct", is_pct=True)
        _extract_yf_field(ratios, fd, "debtToEquity", "debt_to_equity")
        _extract_yf_field(ratios, fd, "currentRatio", "current_ratio")

        # Revenue / cash flow big numbers
        for src_key, dst_key in [
            ("totalRevenue", "total_revenue"), ("totalCash", "total_cash"),
            ("totalDebt", "total_debt"), ("freeCashflow", "free_cashflow"),
            ("operatingCashflow", "operating_cashflow"),
        ]:
            v = _raw(fd, src_key)
            if v is not None:
                result[dst_key] = int(v)

        # Key statistics
        ks = r.get("defaultKeyStatistics", {})
        _extract_yf_field(ratios, ks, "priceToBook", "pb_ratio")  # prefer ks over sd
        _extract_yf_field(ratios, ks, "enterpriseToRevenue", "ev_to_revenue")
        _extract_yf_field(ratios, ks, "enterpriseToEbitda", "ev_to_ebitda")
        shares = _raw(ks, "sharesOutstanding")
        if shares:
            result["shares_outstanding"] = int(shares)

        # Description
        ap = r.get("assetProfile", {})
        if ap.get("longBusinessSummary"):
            result["description"] = ap["longBusinessSummary"]
        if ap.get("sector"):
            result["sector"] = ap["sector"]
        if ap.get("industry"):
            result["industry"] = ap["industry"]
        if ap.get("fullTimeEmployees"):
            result["employees"] = ap["fullTimeEmployees"]

        if ratios:
            result["ratios"] = ratios

        # Price history via v8 chart API
        try:
            chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            chart_params = {"range": "1y", "interval": "1d"}
            with httpx.Client(timeout=10, headers=headers) as client2:
                chart_resp = client2.get(chart_url, params=chart_params)
                if chart_resp.status_code == 200:
                    chart_data = chart_resp.json()
                    chart_result = chart_data.get("chart", {}).get("result", [])
                    if chart_result:
                        closes = chart_result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
                        closes = [c for c in closes if c is not None]
                        if closes:
                            result["price"] = {
                                "current": round(float(closes[-1]), 2),
                                "high_52w": round(float(max(closes)), 2),
                                "low_52w": round(float(min(closes)), 2),
                            }
        except Exception:
            pass

        return result, None
    except Exception as e:
        return None, f"yahoo_v10 {type(e).__name__}: {e}"


def _raw(obj: dict, key: str):
    """Extract .raw value from a Yahoo Finance nested dict, or None."""
    v = obj.get(key)
    if isinstance(v, dict):
        return v.get("raw")
    return v


def _extract_yf_field(ratios: dict, source: dict, src_key: str, dst_key: str,
                      is_pct: bool = False):
    v = _raw(source, src_key)
    if v is not None:
        ratios[dst_key] = round(float(v) * 100 if is_pct else float(v), 2)


# ── Source 6: akshare (East Money US/HK financials) ──

def _try_akshare_cn(ticker: str, code: str) -> tuple:
    """Fetch comprehensive quarterly financial data for A-shares via akshare.

    Uses stock_financial_abstract which returns 80+ indicators across 100+
    quarterly periods (1998 to latest quarter). One API call covers all history.
    Returns (data_dict, error_reason).
    """
    try:
        import akshare as ak
    except ImportError:
        return None, "akshare not installed"

    import pandas as pd

    try:
        df = ak.stock_financial_abstract(symbol=code)
    except Exception as e:
        return None, f"akshare financial_abstract failed: {type(e).__name__}: {e}"

    if df is None or df.empty:
        return None, "akshare financial_abstract returned empty"

    # The DataFrame has rows = indicators, columns = dates (from col 2 onwards)
    # Build a lookup: indicator_name -> {date: value}
    indicators = {}
    date_cols = [c for c in df.columns if c not in ("选项", "指标")]

    for _, row in df.iterrows():
        name = str(row["指标"]).strip()
        vals = {}
        for dc in date_cols:
            v = row[dc]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            try:
                vals[str(dc)] = round(float(v), 2)
            except (ValueError, TypeError):
                pass
        if vals:
            indicators[name] = vals

    if not indicators:
        return None, "akshare financial_abstract: no valid indicator data"

    # Latest date for backward-compatible top-level fields
    latest_date = max(date_cols)

    def _find_key(*keywords):
        """Fuzzy-find an indicator key by required keywords (all must match)."""
        for key in indicators:
            if all(kw in key for kw in keywords):
                return key
        return None

    def _get_val(indicator_key):
        """Get the latest value for a resolved indicator key."""
        if indicator_key is None:
            return None
        vals = indicators.get(indicator_key, {})
        v = vals.get(latest_date)
        return v

    def _get_vals(indicator_key):
        """Get ALL date values for a resolved indicator key."""
        if indicator_key is None:
            return {}
        return indicators.get(indicator_key, {})

    # Resolve indicator keys by keyword matching
    key_revenue = _find_key("营业总", "收入") or _find_key("营业", "收入")
    key_cost = _find_key("营业", "成本")
    key_net_income = _find_key("归母", "净") or _find_key("净利", "润")
    key_roe = _find_key("ROE") or _find_key("净资产收益")
    key_gross_margin = _find_key("毛利", "率")
    key_profit_margin = _find_key("销售净利") or _find_key("净利", "率")
    key_debt_ratio = _find_key("资产负债", "率")
    key_eps = _find_key("基本每股", "收益")
    key_bps = _find_key("每股净", "资产")
    key_ocf = _find_key("经营", "现金", "净") or _find_key("经营", "现金", "流")
    key_rev_growth = _find_key("营业总", "收入", "增长") or _find_key("营业", "收入", "增长")
    key_ni_growth = _find_key("归母", "净利", "增长") or _find_key("归属", "净利", "增长")
    key_current_ratio = _find_key("流动", "比率")
    key_quick_ratio = _find_key("速动", "比率")
    key_equity = _find_key("股东权益") or _find_key("净资产", "合计")

    revenue = _get_val(key_revenue)
    net_income = _get_val(key_net_income)
    cost = _get_val(key_cost)
    gross_profit = round(revenue - cost, 2) if (revenue is not None and cost is not None) else None

    result = {
        "ticker": ticker,
        "data_quality": "akshare",
        "data_source": "akshare",
        "name": ticker,
    }

    if revenue:
        result["total_revenue"] = int(revenue)
        result["revenue"] = int(revenue)
    if net_income:
        result["net_income"] = int(net_income)
    if gross_profit:
        result["gross_profit"] = int(gross_profit)
    oc = _get_val(key_ocf)
    if oc:
        result["operating_cashflow"] = int(oc)

    # Latest ratios (backward compatible)
    ratios = {}
    for ik, our_key in [
        (key_roe, "roe_pct"),
        (key_gross_margin, "gross_margin_pct"),
        (key_profit_margin, "profit_margin_pct"),
        (key_debt_ratio, "debt_ratio_pct"),
        (key_rev_growth, "revenue_growth_pct"),
        (key_current_ratio, "current_ratio"),
        (key_quick_ratio, "quick_ratio"),
    ]:
        v = _get_val(ik)
        if v is not None:
            ratios[our_key] = v
    eps_v = _get_val(key_eps)
    if eps_v is not None:
        ratios["eps"] = eps_v
    bv = _get_val(key_bps)
    if bv is not None:
        ratios["book_value_per_share"] = bv
    if ratios:
        result["ratios"] = ratios

    result["report_date"] = latest_date

    # ── Build multi-year income statement ──
    annual_dates = sorted([d for d in date_cols if d.endswith("1231")], reverse=True)
    if not annual_dates:
        annual_dates = sorted(date_cols, reverse=True)

    income_stmt = {}
    rev_vals = _get_vals(key_revenue)
    cost_vals = _get_vals(key_cost)
    ni_vals = _get_vals(key_net_income)
    gm_vals = _get_vals(key_gross_margin)

    for d in annual_dates:
        r = rev_vals.get(d)
        if r is not None:
            income_stmt.setdefault("Total Revenue", {})[d] = r
        c = cost_vals.get(d)
        if c is not None:
            income_stmt.setdefault("Cost of Revenue", {})[d] = c
        gp = None
        if r is not None and c is not None:
            gp = round(r - c, 2)
        elif gm_vals and r is not None:
            gm = gm_vals.get(d)
            if gm is not None:
                gp = round(r * gm / 100, 2)
        if gp is not None:
            income_stmt.setdefault("Gross Profit", {})[d] = gp
        n = ni_vals.get(d)
        if n is not None:
            income_stmt.setdefault("Net Income", {})[d] = n

    if income_stmt:
        result["financials"] = {"income": income_stmt}

    # Operating cashflow
    ocf_vals = _get_vals(key_ocf)
    if ocf_vals:
        cf_stmt = {"Operating Cash Flow": {d: ocf_vals[d] for d in annual_dates if d in ocf_vals}}
        if cf_stmt["Operating Cash Flow"]:
            result["financials"]["cashflow"] = cf_stmt

    # Balance sheet summary
    equity_vals = _get_vals(key_equity)
    debt_vals = _get_vals(key_debt_ratio)
    if equity_vals:
        bs = {"Total Equity": {d: equity_vals[d] for d in annual_dates if d in equity_vals}}
        if bs["Total Equity"]:
            result["financials"]["balance"] = bs

    # ── Multi-year metrics history ──
    metrics_history = []
    for d in annual_dates[:10]:
        entry = {"period": d}
        r = rev_vals.get(d)
        if r is not None:
            entry["revenue"] = int(r)
        n = ni_vals.get(d)
        if n is not None:
            entry["net_income"] = int(n)
        for ik, our_key in [
            (key_roe, "roe_pct"),
            (key_gross_margin, "gross_margin_pct"),
            (key_profit_margin, "profit_margin_pct"),
            (key_debt_ratio, "debt_ratio_pct"),
            (key_rev_growth, "revenue_growth_pct"),
            (key_current_ratio, "current_ratio"),
        ]:
            vals = _get_vals(ik)
            v = vals.get(d)
            if v is not None:
                entry[our_key] = v
        eps_vals = _get_vals(key_eps)
        if d in eps_vals:
            entry["eps"] = eps_vals[d]
        metrics_history.append(entry)

    if metrics_history:
        result["metrics_history"] = metrics_history

    # ── Latest quarters (年报口径之外的高频数据 — 周期拐点在这里) ──
    quarter_dates = sorted([d for d in date_cols if not d.endswith("1231")], reverse=True)[:4]
    rev_growth_vals = _get_vals(key_rev_growth)
    ni_growth_vals = _get_vals(key_ni_growth)
    quarterly = []
    for d in quarter_dates:
        entry = {"period": d}
        r, n = rev_vals.get(d), ni_vals.get(d)
        if r is not None:
            entry["revenue"] = int(r)
        if n is not None:
            entry["net_income"] = int(n)
        rg, ng = rev_growth_vals.get(d), ni_growth_vals.get(d)
        if rg is not None:
            entry["revenue_yoy_pct"] = rg
        if ng is not None:
            entry["profit_yoy_pct"] = ng
        if len(entry) > 1:
            quarterly.append(entry)
    if quarterly:
        result["quarterly"] = quarterly

    return result, None


def try_akshare(ticker: str) -> tuple:
    """Fetch financial statements for US/HK stocks via akshare (East Money backend).

    Returns ALL available fiscal years (not just the latest), building multi-year
    income statements and ratio history. Data is current and covers 5-15 years.
    Returns (data_dict, error_reason).
    """
    try:
        import akshare as ak
    except ImportError:
        return None, "akshare not installed (pip install akshare)"

    t = ticker.upper()
    is_hk = t.endswith(".HK")
    is_cn = t.endswith((".SS", ".SZ", ".SH", ".BJ"))
    is_us = not is_hk and not is_cn

    try:
        if is_hk:
            code = t.split(".")[0].zfill(5)
            df = ak.stock_financial_hk_analysis_indicator_em(symbol=code)
        elif is_cn:
            code = t.split(".")[0]
            return _try_akshare_cn(ticker, code)
        elif is_us:
            df = ak.stock_financial_us_analysis_indicator_em(symbol=ticker)
        else:
            return None, "akshare not applicable for A-shares"
    except Exception as e:
        return None, f"akshare fetch failed: {type(e).__name__}: {e}"

    if df is None or df.empty:
        return None, "akshare returned empty data"

    import pandas as pd

    # Map net income field name (US uses PARENT_HOLDER_NETPROFIT, HK uses HOLDER_PROFIT)
    net_income_col = "HOLDER_PROFIT" if is_hk else "PARENT_HOLDER_NETPROFIT"

    # Field mapping: (akshare_key, our_key, is_pct)
    _RATIO_MAP = [
        ("ROE_AVG", "roe_pct", True),
        ("GROSS_PROFIT_RATIO", "gross_margin_pct", True),
        ("NET_PROFIT_RATIO", "profit_margin_pct", True),
        ("DEBT_ASSET_RATIO", "debt_ratio_pct", True),
        ("OPERATE_INCOME_YOY", "revenue_growth_pct", True),
        ("CURRENT_RATIO", "current_ratio", False),
    ]

    # Latest row (in original order) for backward-compatible top-level fields
    latest = df.iloc[0]
    def _af(key, default=None):
        val = latest.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        if isinstance(val, (int, float)):
            return round(float(val), 2)
        return val

    result = {
        "ticker": ticker,
        "data_quality": "akshare",
        "data_source": "akshare",
        "name": _af("SECURITY_NAME_ABBR") or ticker,
    }

    # Backward-compatible top-level fields (latest year)
    revenue = _af("OPERATE_INCOME")
    net_income = _af(net_income_col)
    gross_profit = _af("GROSS_PROFIT")

    if revenue:
        result["total_revenue"] = int(revenue)
        result["revenue"] = int(revenue)
    if net_income:
        result["net_income"] = int(net_income)
    if gross_profit:
        result["gross_profit"] = int(gross_profit)

    # Latest year ratios (backward compatible)
    ratios = {}
    for ak_key, our_key, _ in _RATIO_MAP:
        v = _af(ak_key)
        if v is not None:
            ratios[our_key] = v
    bps = _af("BPS")
    eps = _af("BASIC_EPS")
    if bps:
        ratios["book_value_per_share"] = bps
    if eps:
        ratios["eps"] = eps
    if ratios:
        result["ratios"] = ratios

    report_date = latest.get("REPORT_DATE")
    if report_date:
        result["report_date"] = str(report_date)[:10]

    # ── Build multi-year data from ALL rows ──
    income_stmt = {}
    metrics_history = []

    for i in range(len(df)):
        row = df.iloc[i]
        rd = row.get("REPORT_DATE")
        if rd is None:
            continue
        period = str(rd)[:10]

        def _rv(col):
            """Read a value from the row, return rounded float or None."""
            val = row.get(col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return round(float(val), 2)

        # Income statement entries for this period
        r = _rv("OPERATE_INCOME")
        gp = _rv("GROSS_PROFIT")
        ni = _rv(net_income_col)
        if r is not None:
            income_stmt.setdefault("Total Revenue", {})[period] = r
        if gp is not None:
            income_stmt.setdefault("Gross Profit", {})[period] = gp
        if ni is not None:
            income_stmt.setdefault("Net Income", {})[period] = ni

        # Key metrics for this period
        entry = {"period": period}
        if r is not None:
            entry["revenue"] = int(r)
        if ni is not None:
            entry["net_income"] = int(ni)
        for ak_key, our_key, _ in _RATIO_MAP:
            v = _rv(ak_key)
            if v is not None:
                entry[our_key] = v
        eps_v = _rv("BASIC_EPS")
        if eps_v is not None:
            entry["eps"] = eps_v
        metrics_history.append(entry)

    if income_stmt:
        result["financials"] = {"income": income_stmt}
    if metrics_history:
        result["metrics_history"] = metrics_history

    # Also get quarterly data for the latest 3 years via stock_yjbb_em for A-shares
    # (US/HK already have annual data above; this supplements with quarterly detail)

    return result, None


# ── Fresh news headlines (情绪分析的真实语料，防止 LLM 用过期训练知识编叙事) ──

def fetch_news(ticker: str, limit: int = 8) -> list:
    """Fetch latest dated news headlines for the stock via East Money (akshare).

    Works for A-shares (600519), HK (00700), US (BABA). Returns
    [{"datetime": str, "title": str, "source": str}], newest first. [] on failure.
    """
    try:
        import akshare as ak
    except ImportError:
        return []
    t = ticker.upper()
    if t.endswith(".HK"):
        code = t.split(".")[0].zfill(5)
    elif t.endswith((".SS", ".SZ", ".SH", ".BJ")):
        code = t.split(".")[0]
    else:
        code = t
    try:
        df = ak.stock_news_em(symbol=code)
        if df is None or df.empty:
            return []
        out = []
        for _, row in df.head(limit).iterrows():
            title = str(row.get("新闻标题") or "").strip()
            if not title:
                continue
            out.append({
                "datetime": str(row.get("发布时间") or "")[:16],
                "title": title,
                "source": str(row.get("文章来源") or "").strip(),
            })
        return out
    except Exception:
        return []


# ── Valuation anchors (行业/同行/自身历史 — 防止 LLM 自编对比标的) ──

def fetch_valuation_anchor(ticker: str) -> dict:
    """Fetch real valuation anchors: industry median/average multiples, peer table,
    and own historical PE percentile.

    A-shares: EM peer comparison (real peer list incl. forward PE) + baidu 10y PE band.
    HK: EM industry rank of own multiples. US: unavailable ({}).
    """
    t = ticker.upper()
    try:
        if t.endswith((".SS", ".SZ", ".SH", ".BJ")):
            return _anchor_cn(t)
        if t.endswith(".HK"):
            return _anchor_hk(t)
    except Exception:
        pass
    return {}


def _anchor_cn(t: str) -> dict:
    import akshare as ak
    code, suffix = t.split(".")
    em_symbol = ("SH" if suffix in ("SS", "SH") else "SZ") + code

    out = {"anchor_source": "eastmoney"}
    try:
        df = ak.stock_zh_valuation_comparison_em(symbol=em_symbol)
    except Exception:
        df = None
    if df is not None and not df.empty:
        keep_cols = [c for c in df.columns
                     if c in ("代码", "简称", "PEG") or c.startswith(("市盈率-", "市净率-", "市销率-"))]
        peers = []
        for _, row in df.iterrows():
            name = str(row.get("简称") or "")
            entry = {}
            for c in keep_cols:
                v = row.get(c)
                try:
                    import math
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        continue
                    entry[c] = round(float(v), 2) if c not in ("代码", "简称") else v
                except (TypeError, ValueError):
                    continue
            if not name:
                continue
            if str(row.get("代码") or "") == code:
                out["self"] = entry
            elif name in ("行业中值", "行业平均"):
                out["industry_median" if name == "行业中值" else "industry_average"] = entry
            elif len(peers) < 5:
                peers.append(entry)
        if peers:
            out["peers"] = peers

    # Own 10-year PE-TTM percentile (best-effort, baidu source is flaky)
    try:
        band = ak.stock_zh_valuation_baidu(symbol=code, indicator="市盈率(TTM)", period="近十年")
        if band is not None and not band.empty:
            vals = band.iloc[:, -1].astype(float)
            vals = vals[vals > 0]
            if len(vals) > 20:
                cur = vals.iloc[-1]
                out["pe_ttm_current"] = round(float(cur), 2)
                out["pe_ttm_percentile_10y"] = round(float((vals <= cur).mean() * 100), 1)
    except Exception:
        pass

    return out if len(out) > 1 else {}


def _anchor_hk(t: str) -> dict:
    import akshare as ak
    code = t.split(".")[0].zfill(5)
    try:
        df = ak.stock_hk_valuation_comparison_em(symbol=code)
    except Exception:
        return {}
    if df is None or df.empty:
        return {}
    row = df.iloc[0]
    out = {"anchor_source": "eastmoney"}
    for col, key in [("市盈率-TTM", "pe_ttm"), ("市盈率-TTM排名", "pe_ttm_industry_rank"),
                     ("市净率-MRQ", "pb"), ("市净率-MRQ排名", "pb_industry_rank"),
                     ("市销率-TTM", "ps_ttm")]:
        v = _num(row.get(col))
        if v is not None:
            out[key] = int(v) if "排名" in col else round(v, 2)
    return out if len(out) > 1 else {}


# ── Analyst consensus estimates (盈利预测/一致预期) ──
# Forward-looking REAL data: what analysts forecast for future fiscal years.
# Without this, future years in the report are pure model speculation.

def fetch_estimates(ticker: str) -> dict:
    """Fetch analyst consensus estimates for future fiscal years.

    Returns {} when unavailable. Otherwise:
    {
      "est_source": "eastmoney|ths|etnet|yfinance",
      "currency": "CNY|HKD|USD",
      "analyst_count": int | None,
      "years": [{"year": 2026, "mark": "E", "eps": 68.9,
                 "eps_min": .., "eps_max": .., "revenue": .., "net_profit": ..}],
      "target_price": {"min": .., "max": ..} | None,
      "ratings": {"buy": .., "add": .., "neutral": .., "reduce": .., "sell": ..} | None,
    }
    mark: "A" = already reported actual, "E" = analyst estimate.
    """
    t = ticker.upper()
    try:
        if t.endswith((".SS", ".SZ", ".SH", ".BJ")):
            return _estimates_cn(t) or _estimates_cn_ths(t)
        if t.endswith(".HK"):
            return _estimates_hk(t)
        return _estimates_us(t)
    except Exception:
        return {}


def _estimates_cn(t: str) -> dict:
    """A-shares: East Money F10 consensus (RPT_WEB_RESPREDICT)."""
    secucode = _em_secucode(t)
    if not secucode:
        return {}
    try:
        url = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
        params = {
            "reportName": "RPT_WEB_RESPREDICT",
            "columns": "ALL",
            "filter": f'(SECUCODE="{secucode}")',
            "pageNumber": 1, "pageSize": 1,
            "source": "HSF10", "client": "PC",
        }
        with httpx.Client(timeout=10, headers=_HEADERS) as client:
            payload = client.get(url, params=params).json()
        rows = (payload.get("result") or {}).get("data") or []
        if not rows:
            return {}
        row = rows[0]

        years = []
        for i in range(1, 5):
            yr, mark, eps = row.get(f"YEAR{i}"), row.get(f"YEAR_MARK{i}"), _num(row.get(f"EPS{i}"))
            if yr is None or eps is None:
                continue
            years.append({"year": int(yr), "mark": mark or "E", "eps": round(eps, 2)})
        if not years:
            return {}

        ratings = {}
        for k, raw in [("buy", "RATING_BUY_NUM"), ("add", "RATING_ADD_NUM"),
                       ("neutral", "RATING_NEUTRAL_NUM"), ("reduce", "RATING_REDUCE_NUM"),
                       ("sell", "RATING_SALE_NUM")]:
            v = _num(row.get(raw))
            if v:
                ratings[k] = int(v)

        tp_min, tp_max = _num(row.get("DEC_AIMPRICEMIN")), _num(row.get("DEC_AIMPRICEMAX"))
        out = {
            "est_source": "eastmoney",
            "currency": "CNY",
            "analyst_count": int(_num(row.get("RATING_ORG_NUM")) or 0) or None,
            "years": years,
        }
        if ratings:
            out["ratings"] = ratings
        if tp_min or tp_max:
            out["target_price"] = {"min": tp_min, "max": tp_max}
        return out
    except Exception:
        return {}


def _estimates_cn_ths(t: str) -> dict:
    """A-shares fallback: THS consensus EPS via akshare."""
    try:
        import akshare as ak
        df = ak.stock_profit_forecast_ths(symbol=t.split(".")[0], indicator="预测年报每股收益")
        if df is None or df.empty:
            return {}
        years = []
        for _, r in df.iterrows():
            eps = _num(r.get("均值"))
            if eps is None:
                continue
            entry = {"year": int(r.get("年度")), "mark": "E", "eps": round(eps, 2)}
            lo, hi = _num(r.get("最小值")), _num(r.get("最大值"))
            if lo is not None:
                entry["eps_min"] = round(lo, 2)
            if hi is not None:
                entry["eps_max"] = round(hi, 2)
            years.append(entry)
        if not years:
            return {}
        cnt = _num(df.iloc[0].get("预测机构数"))
        return {"est_source": "ths", "currency": "CNY",
                "analyst_count": int(cnt) if cnt else None, "years": years}
    except Exception:
        return {}


def _estimates_hk(t: str) -> dict:
    """HK: etnet consensus via akshare (aggregated + per-broker detail)."""
    try:
        import akshare as ak
        code = t.split(".")[0].zfill(5)
        df = ak.stock_hk_profit_forecast_et(symbol=code, indicator="综合盈利预测")
        if df is None or df.empty:
            return {}
        years = []
        for _, r in df.iterrows():
            fy = _num(r.get("财政年度"))
            eps_c = _num(r.get("每股盈利/每股亏损"))
            np_m = _num(r.get("纯利/亏损"))
            if fy is None:
                continue
            entry = {"year": int(fy), "mark": "E"}
            if eps_c is not None:
                entry["eps"] = round(eps_c / 100, 2)  # cents -> HKD
            if np_m is not None:
                entry["net_profit"] = int(np_m * 1e6)  # millions -> HKD
            if entry.get("eps") is not None or entry.get("net_profit") is not None:
                years.append(entry)
        if not years:
            return {}
        out = {"est_source": "etnet", "currency": "HKD",
               "analyst_count": None, "years": years}
        # Best-effort: per-broker detail for target price + broker count (flaky endpoint)
        try:
            det = ak.stock_hk_profit_forecast_et(symbol=code, indicator="盈利预测明细")
            if det is not None and not det.empty:
                tps = [_num(v) for v in det.get("目标价", [])]
                tps = [v for v in tps if v]
                brokers = det.get("证券商", [])
                if tps:
                    tps_sorted = sorted(tps)
                    mid = tps_sorted[len(tps_sorted) // 2]
                    out["target_price"] = {"min": min(tps), "max": max(tps), "median": mid}
                if len(brokers):
                    out["analyst_count"] = len(set(brokers))
        except Exception:
            pass
        return out
    except Exception:
        return {}


def _estimates_us(t: str) -> dict:
    """US: yfinance analyst estimates (subject to Yahoo rate limits)."""
    global _yf_circuit_open_until
    if _time.time() < _yf_circuit_open_until:
        return {}
    try:
        import yfinance as yf
        stock = yf.Ticker(t)
        est = stock.get_earnings_estimate()
        if est is None or est.empty:
            return {}
        rev, growth = None, None
        try:
            rev = stock.get_revenue_estimate()
        except Exception:
            pass
        try:
            growth = stock.get_growth_estimates()
        except Exception:
            pass

        period_labels = {"0y": "当前财年", "+1y": "下一财年"}
        years = []
        for p in ("0y", "+1y"):
            if p not in est.index:
                continue
            r = est.loc[p]
            eps = _num(r.get("avg"))
            if eps is None:
                continue
            entry = {"year": period_labels[p], "mark": "E", "eps": round(eps, 2)}
            lo, hi = _num(r.get("low")), _num(r.get("high"))
            if lo is not None:
                entry["eps_min"] = round(lo, 2)
            if hi is not None:
                entry["eps_max"] = round(hi, 2)
            if rev is not None and p in rev.index:
                rv = _num(rev.loc[p].get("avg"))
                if rv is not None:
                    entry["revenue"] = int(rv)
            if growth is not None and p in growth.index:
                g = _num(growth.loc[p].get("stockTrend"))
                if g is not None:
                    entry["eps_growth_pct"] = round(g * 100, 2)
            cnt = _num(r.get("numberOfAnalysts"))
            if cnt:
                entry["analyst_count"] = int(cnt)
            years.append(entry)
        if not years:
            return {}
        counts = [e.get("analyst_count") for e in years if e.get("analyst_count")]
        return {"est_source": "yfinance", "currency": "USD",
                "analyst_count": max(counts) if counts else None, "years": years}
    except Exception as e:
        if _is_rate_limit(e):
            _yf_circuit_open_until = _time.time() + 60
        return {}


# ── Source 7: LLM Knowledge Fallback (OFF by default) ──

LLM_FALLBACK_PROMPT = """You are a financial data provider. Based on your training knowledge, provide accurate financial data for {ticker} ({company_name}).

Return ONLY a valid JSON object. Use null for any value you are not confident about:

{{
  "name": "full company name",
  "sector": "sector",
  "industry": "industry",
  "description": "one-paragraph business description",
  "market_cap": estimated market cap in USD (integer or null),
  "current_price": estimated stock price in USD (number or null),
  "pe_ratio": trailing P/E (number or null),
  "forward_pe": forward P/E (number or null),
  "pb_ratio": price-to-book (number or null),
  "ps_ratio": price-to-sales (number or null),
  "debt_to_equity": D/E ratio (number or null),
  "roe_pct": ROE as percentage (number or null),
  "profit_margin_pct": net margin as percentage (number or null),
  "revenue_growth_pct": revenue growth as percentage (number or null),
  "beta": beta (number or null),
  "dividend_yield_pct": dividend yield as percentage (number or null),
  "revenue_billions": annual revenue in billions USD (number or null),
  "net_income_billions": annual net income in billions USD (number or null),
  "free_cash_flow_billions": FCF in billions USD (number or null),
  "employees": employee count (integer or null),
  "exchange": "exchange name",
  "currency": "currency code",
  "country": "country"
}}"""


def try_llm_knowledge(ticker: str, company_name: str, llm_client, model: str) -> dict:
    """Fetch financial data from LLM training knowledge (STALE — opt-in only)."""
    prompt = LLM_FALLBACK_PROMPT.format(ticker=ticker, company_name=company_name or ticker)
    try:
        text = chat(llm_client, model=model, max_tokens=2048, temperature=0.1,
                    messages=[{"role": "user", "content": prompt}])
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        data = json.loads(text.strip())
        data["ticker"] = ticker
        data["data_quality"] = "llm_knowledge"
        data["data_source"] = "llm_knowledge"
        return data
    except Exception as e:
        return {"ticker": ticker, "data_quality": "failed", "error": str(e)}


# ── Orchestrated fetch ──

def _source_order(ticker: str) -> list:
    t = ticker.upper()
    if t.endswith((".SS", ".SZ", ".SH", ".BJ")):
        # A-shares: akshare first (financial_abstract gives full quarterly data)
        return [try_akshare, try_eastmoney, try_tencent, try_sina, try_yfinance]
    if t.endswith(".HK"):
        return [try_eastmoney, try_tencent, try_akshare, try_yahoo_v10, try_yfinance, try_sina]
    # US / other: yfinance → akshare → yahoo_v10 → eastmoney → tencent → sina
    return [try_yfinance, try_akshare, try_yahoo_v10, try_eastmoney, try_tencent, try_sina]


def fetch_all(ticker: str, llm_client=None, llm_model=None, company_hint: str = "",
              allow_llm_fallback: bool = False) -> dict:
    """Try data sources in market-aware order. Returns the first successful result.

    LLM knowledge fallback is OFF by default. Set allow_llm_fallback=True to
    enable it when all external sources fail (data will be ~1-2 years stale).
    """
    sources = []
    errors = []
    best_result = None

    for source in _source_order(ticker):
        name = source.__name__.replace("try_", "")
        result, err = source(ticker)
        if result is not None:
            # Merge financial data from multiple sources when possible.
            # The FIRST source that returns price wins, but we supplement
            # with financial ratios from later sources if the first is quote-only.
            if best_result is None:
                best_result = result
                # Supplement with A-share F10 financials if missing
                if not best_result.get("financials"):
                    f10 = _em_f10_financials(ticker)
                    if f10:
                        best_result.setdefault("ratios", {}).update(f10["ratios"])
                        best_result["financials"] = f10["financials"]
                        best_result["data_quality"] = f"{name}+em_f10"
            else:
                # Merge ratios/financials from richer source into best_result
                if result.get("ratios"):
                    if not best_result.get("ratios"):
                        best_result["ratios"] = dict(result["ratios"])
                    else:
                        # Merge additional ratios (tencent only has PE/PB, akshare has 7+ ratios)
                        for k, v in result["ratios"].items():
                            if v is not None and k not in best_result["ratios"]:
                                best_result["ratios"][k] = v
                    best_result["data_quality"] = best_result.get("data_quality", "") + f"+{name}_ratios"
                for k in ("total_revenue", "total_cash", "total_debt",
                          "free_cashflow", "operating_cashflow", "shares_outstanding",
                          "sector", "industry", "description", "employees",
                          "revenue", "net_income", "gross_profit",
                          "report_date", "fiscal_year"):
                    if result.get(k) is not None and best_result.get(k) is None:
                        best_result[k] = result[k]
                # Merge financial statements
                if result.get("financials") and not best_result.get("financials"):
                    best_result["financials"] = result["financials"]
                # Merge basic quote fields from quote sources into financial sources
                for k in ("current_price", "market_cap", "currency",
                          "price", "exchange", "country"):
                    if result.get(k) is not None and best_result.get(k) is None:
                        best_result[k] = result[k]
                # Merge name if better (English name from Tencent is better than Chinese from akshare)
                if result.get("name") and (not best_result.get("name") or
                   (best_result.get("data_source") == "akshare" and result.get("data_source") == "tencent")):
                    # Prefer English names from Tencent over akshare's Chinese names
                    rname = result["name"]
                    bname = best_result["name"]
                    if any('一' <= c <= '鿿' for c in bname) and not any('一' <= c <= '鿿' for c in rname):
                        best_result["name"] = rname
            sources.append(f"{name}:ok")
        else:
            sources.append(f"{name}:failed")
            if err:
                errors.append(f"  {name}: {err}")

    if best_result is not None:
        # Attach analyst consensus estimates (forward-looking real data) when available
        est = fetch_estimates(ticker)
        if est:
            best_result["estimates"] = est
            src = est.get("est_source", "?")
            n = len(est.get("years", []))
            best_result["data_quality"] = f"{best_result.get('data_quality', '')}+{src}_estimates"
            sources.append(f"{src}_estimates:ok({n}y)")
        else:
            sources.append("estimates:unavailable")
        # Attach fresh news headlines (sentiment grounding) and valuation anchors
        news = fetch_news(ticker)
        if news:
            best_result["news"] = news
            sources.append(f"news:ok({len(news)})")
        anchor = fetch_valuation_anchor(ticker)
        if anchor:
            best_result["valuation_anchor"] = anchor
            sources.append("valuation_anchor:ok")
        best_result["_sources_tried"] = sources
        return best_result

    # All external sources failed
    print(f"\n{'─'*50}")
    print(f"  [WARN] 所有外部数据源均失败:")
    for e in errors:
        print(e)
    print(f"{'─'*50}")

    if allow_llm_fallback and llm_client:
        print("\n  [!!!] 降级到 LLM 训练知识（约 1-2 年前的静态数据，非实时行情）\n")
        result = try_llm_knowledge(ticker, company_hint, llm_client, llm_model)
        if result.get("data_quality") != "failed":
            result["_sources_tried"] = sources + ["llm_knowledge:fallback"]
            return result
        sources.append("llm_knowledge:failed")

    return {"ticker": ticker, "data_quality": "all_failed",
            "data_source": "none", "_sources_tried": sources,
            "error": "All data sources failed"}
