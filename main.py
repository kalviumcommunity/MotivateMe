import os
import requests
import json
import re
import argparse
import sys
from typing import Optional, Tuple

# Try to import token logger (optional)
try:
    from token_utils import log_tokens_after_call
except Exception:
    def log_tokens_after_call(api_response=None, prompt_text=None, response_text=None, model_name=None):
        def estimate(t): return max(1, int(len(t) / 4)) if t else 0
        pt = estimate(prompt_text or "")
        ct = estimate(response_text or "")
        print(f"[Token estimate] prompt={pt} completion={ct} total={pt+ct}")

# ----------------------
# Helpers
# ----------------------
def strip_code_fences(text: str) -> str:
    """Remove common markdown code fences like ```json or ``` ... ```."""
    if not text:
        return text
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def extract_first_json(text: str) -> Optional[str]:
    """Find the first balanced { ... } JSON substring."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def validate_structured_output(obj: dict) -> Tuple[bool, str]:
    """Ensure required keys are present and are non-empty strings."""
    required = ["mood", "quote", "author", "suggested_action"]
    if not isinstance(obj, dict):
        return False, "Output is not a JSON object"
    for k in required:
        if k not in obj:
            return False, f"Missing key: {k}"
        if not isinstance(obj[k], str) or not obj[k].strip():
            return False, f"Invalid or empty value for key: {k}"
    return True, "OK"

def save_last_output(obj: dict, path: str = "last_output.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[Saved] structured output -> {path}")

# ----------------------
# Google Studio call
# ----------------------
def call_google_studio_structured(prompt_text: str,
                                  api_key: str,
                                  temperature: float = 0.2,
                                  top_k: Optional[int] = None,
                                  top_p: Optional[float] = None,
                                  stop_sequences: Optional[list] = None,
                                  model: str = "gemini-1.5-flash-latest") -> Tuple[dict, str]:
    if not api_key:
        raise RuntimeError("Provide api_key or set GOOGLE_API_KEY env var")

    base = "https://generativelanguage.googleapis.com/v1beta/models"
    url = f"{base}/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    generation_config = {"temperature": float(temperature)}
    if top_k is not None:
        generation_config["top_k"] = int(top_k)
    if top_p is not None:
        generation_config["top_p"] = float(top_p)
    if stop_sequences:
        generation_config["stopSequences"] = list(stop_sequences)

    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": generation_config
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    api_response = resp.json()

    try:
        response_text = api_response["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        response_text = json.dumps(api_response, ensure_ascii=False)

    return api_response, response_text

# ----------------------
# High-level function
# ----------------------
def get_structured_quote(user_mood: str,
                         api_key: str,
                         temperature: float = 0.2,
                         top_k: Optional[int] = None,
                         top_p: Optional[float] = None) -> Optional[dict]:

    stop_marker = "<END_JSON>"
    system_instr = (
        "You are a motivational quote generator. "
        "Respond ONLY with a valid JSON object containing exactly these keys: "
        "mood, quote, author, suggested_action. "
        "No markdown, no code fences, no explanation. "
        f"End your output with {stop_marker}."
    )

    user_line = f'User mood: "{user_mood}".'

    prompt = f"{system_instr}\n{user_line}\nOutput the JSON now:\n"

    api_resp, raw = call_google_studio_structured(
        prompt_text=prompt,
        api_key=api_key,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_sequences=[stop_marker]
    )

    log_tokens_after_call(api_response=api_resp, prompt_text=prompt, response_text=raw, model_name="gemini")

    cleaned = strip_code_fences(raw)
    parsed = None

    # Try direct parse
    try:
        parsed = json.loads(cleaned)
    except Exception:
        # Try extraction
        candidate = extract_first_json(cleaned)
        if candidate:
            try:
                parsed = json.loads(candidate)
            except Exception:
                parsed = None

    # Retry once if failed
    if parsed is None:
        print("[Retry] Parsing failed. Requesting again...")
        api_resp, raw = call_google_studio_structured(
            prompt_text=prompt,
            api_key=api_key,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_sequences=[stop_marker]
        )
        cleaned = strip_code_fences(raw)
        try:
            parsed = json.loads(cleaned)
        except Exception:
            candidate = extract_first_json(cleaned)
            if candidate:
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    parsed = None

    if parsed is None:
        print("[Error] Could not parse JSON from model output:\n", raw)
        return None

    ok, msg = validate_structured_output(parsed)
    if not ok:
        print(f"[Validation Failed] {msg}")
        return None

    save_last_output(parsed)
    return parsed

# ----------------------
# CLI
# ----------------------
def main_cli():
    ap = argparse.ArgumentParser(description="Daily Motivation Bot - structured output demo")
    ap.add_argument("--mood", type=str, default="tired", help="User mood string")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--api_key", type=str, default=None, help="Google API key (or set GOOGLE_API_KEY env var)")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Set GOOGLE_API_KEY environment variable or pass --api_key")
        sys.exit(1)

    parsed = get_structured_quote(
        user_mood=args.mood,
        api_key=api_key,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    if parsed:
        print("\n[Structured Output Parsed]")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    else:
        print("\n[Failed to get clean structured output]")

if __name__ == "__main__":
    main_cli()
