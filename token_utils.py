# token_utils.py
import json
import logging
from typing import Optional

# Attempt to import tiktoken for accurate local token counts
try:
    import tiktoken
    _HAS_TIK = True
except Exception:
    _HAS_TIK = False

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("tokens")

def pretty_print_usage(usage: dict):
    """If provider returns usage dict similar to OpenAI, print it."""
    if not usage:
        return
    # Common keys: prompt_tokens, completion_tokens, total_tokens
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    tt = usage.get("total_tokens")
    if pt is not None or ct is not None or tt is not None:
        logger.info(f"Token usage (from API): prompt={pt} completion={ct} total={tt}")
        return
    # Generic fallback: print whole usage object
    try:
        logger.info("Usage (raw): " + json.dumps(usage))
    except Exception:
        logger.info(f"Usage (raw): {usage}")

def count_tokens_tiktoken(text: str, model: Optional[str] = None) -> int:
    """Count tokens using tiktoken. model is optional; if not provided, use cl100k_base."""
    if not _HAS_TIK:
        raise RuntimeError("tiktoken not installed")
    # choose encoding
    try:
        if model:
            enc = tiktoken.encoding_for_model(model)
        else:
            enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def estimate_tokens_heuristic(text: str) -> int:
    """Very rough fallback: average 4 characters per token."""
    if not text:
        return 0
    chars = len(text)
    # 3.5 or 4 char per token is a common heuristic
    return max(1, int(chars / 4))

def log_tokens_after_call(
    *,
    api_response: Optional[dict] = None,
    prompt_text: Optional[str] = None,
    response_text: Optional[str] = None,
    model_name: Optional[str] = None
):
    """
    Call this after you make an LLM/API call.
    - api_response: full raw response from provider (may contain usage info).
    - prompt_text: the exact text you sent as prompt.
    - response_text: the model's text reply (if you extracted it).
    - model_name: optional model string for tokenizer selection.
    """
    # 1) If API returns usage info, prefer that (most accurate)
    usage = None
    if isinstance(api_response, dict):
        # common field names for usage
        for key in ("usage", "token_usage", "quota", "billing"):
            if key in api_response:
                usage = api_response[key]
                break
        # some providers embed usage in other shapes; try direct keys
        if not usage:
            # e.g., openai-style: response["usage"]
            usage = api_response.get("usage")

    if usage:
        pretty_print_usage(usage)
        return

    # 2) No provider usage available. Try local tokenizer (tiktoken) if present:
    prompt_tokens = None
    response_tokens = None
    total_tokens = None

    if _HAS_TIK:
        try:
            if prompt_text is not None:
                prompt_tokens = count_tokens_tiktoken(prompt_text, model=model_name)
            if response_text is not None:
                response_tokens = count_tokens_tiktoken(response_text, model=model_name)
            if prompt_tokens is not None or response_tokens is not None:
                total_tokens = (prompt_tokens or 0) + (response_tokens or 0)
                logger.info(f"Token estimate (tiktoken): prompt={prompt_tokens} completion={response_tokens} total={total_tokens}")
                return
        except Exception as e:
            logger.info(f"tiktoken counting failed: {e}")

    # 3) Fallback heuristic if tiktoken unavailable
    if prompt_text is not None:
        prompt_tokens = estimate_tokens_heuristic(prompt_text)
    if response_text is not None:
        response_tokens = estimate_tokens_heuristic(response_text)
    total_tokens = (prompt_tokens or 0) + (response_tokens or 0)
    logger.info(f"Token estimate (heuristic): prompt={prompt_tokens} completion={response_tokens} total={total_tokens}")
