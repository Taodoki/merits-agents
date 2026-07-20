"""Provider-agnostic LLM configuration.

Supports both Anthropic SDK and OpenAI SDK.
Set LLM_PROVIDER=openai to use OpenAI SDK (default: anthropic).

Environment variables:
  ANTHROPIC_API_KEY   — API key for Anthropic SDK
  ANTHROPIC_BASE_URL  — Base URL for Anthropic SDK
  OPENAI_API_KEY      — API key for OpenAI SDK
  OPENAI_BASE_URL     — Base URL for OpenAI SDK
  LLM_PROVIDER        — "anthropic" (default) or "openai"
"""

import os
from pathlib import Path


def _load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return {}
    vars = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            vars[key.strip()] = val.strip().strip("\"'")
    return vars


_dotenv = _load_dotenv()


def _get(key: str, alt_keys: list = None, default: str = "") -> str:
    # 1. Primary key from os.environ
    val = os.environ.get(key)
    if val:
        return val
    # 2. Primary key from .env file
    val = _dotenv.get(key)
    if val:
        return val
    # 3. Alt keys from os.environ
    if alt_keys:
        for ak in alt_keys:
            val = os.environ.get(ak)
            if val:
                return val
    # 4. Alt keys from .env file
    if alt_keys:
        for ak in alt_keys:
            val = _dotenv.get(ak)
            if val:
                return val
    return default


# ── Provider selection ──
LLM_PROVIDER = _get("LLM_PROVIDER", default="anthropic").lower()

# ── Anthropic SDK config ──
ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY",
                          alt_keys=["ANTHROPIC_AUTH_TOKEN"],
                          default="PROXY_MANAGED")
ANTHROPIC_BASE_URL = _get("ANTHROPIC_BASE_URL",
                           default="https://api.anthropic.com")

# ── OpenAI SDK config ──
OPENAI_API_KEY = _get("OPENAI_API_KEY",
                       alt_keys=["ANTHROPIC_DIRECT_API_KEY"],
                       default="")
OPENAI_BASE_URL = _get("OPENAI_BASE_URL",
                        default="https://api.openai.com/v1")

# ── Effective config (based on provider) ──
if LLM_PROVIDER == "openai":
    API_KEY = OPENAI_API_KEY
    BASE_URL = OPENAI_BASE_URL
else:
    API_KEY = ANTHROPIC_API_KEY
    BASE_URL = ANTHROPIC_BASE_URL

# ── Model names ──
MODEL_FAST = _get("LLM_MODEL_FAST",
                  alt_keys=["MODEL_FAST", "ANTHROPIC_DEFAULT_SONNET_MODEL"],
                  default="claude-sonnet-4-6")
MODEL_BEST = _get("LLM_MODEL_BEST",
                  alt_keys=["MODEL_BEST", "ANTHROPIC_DEFAULT_OPUS_MODEL"],
                  default="claude-opus-4-8")

DEFAULT_TICKER = "600519.SS"


def check_llm_connection(timeout: float = 8.0):
    """Quick connectivity check. Returns (ok: bool, message: str)."""
    if LLM_PROVIDER == "openai":
        return _check_openai(timeout)
    else:
        return _check_anthropic(timeout)


def _check_anthropic(timeout: float):
    import anthropic
    if not ANTHROPIC_API_KEY or len(ANTHROPIC_API_KEY) < 5:
        return False, "ANTHROPIC_API_KEY 未配置"

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY,
                                      base_url=ANTHROPIC_BASE_URL,
                                      timeout=timeout)
        client.messages.create(
            model=MODEL_FAST,
            max_tokens=1,
            messages=[{"role": "user", "content": "."}],
        )
        return True, f"LLM 连接正常 ({ANTHROPIC_BASE_URL}, anthropic SDK)"
    except anthropic.AuthenticationError:
        return False, f"API key 认证失败 ({ANTHROPIC_BASE_URL})"
    except anthropic.APIConnectionError as e:
        return False, f"无法连接 {ANTHROPIC_BASE_URL}: {e}"
    except Exception as e:
        return False, f"连接测试失败: {type(e).__name__}: {e}"


def _check_openai(timeout: float):
    import openai
    if not OPENAI_API_KEY or len(OPENAI_API_KEY) < 10:
        return False, "OPENAI_API_KEY 未配置，请在 .env 中设置"

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY,
                                base_url=OPENAI_BASE_URL,
                                timeout=timeout)
        client.chat.completions.create(
            model=MODEL_FAST,
            max_tokens=1,
            messages=[{"role": "user", "content": "."}],
        )
        return True, f"LLM 连接正常 ({OPENAI_BASE_URL}, openai SDK)"
    except openai.AuthenticationError:
        return False, f"API key 认证失败 ({OPENAI_BASE_URL})"
    except openai.APIConnectionError as e:
        return False, f"无法连接 {OPENAI_BASE_URL}: {e}"
    except Exception as e:
        return False, f"连接测试失败: {type(e).__name__}: {e}"
