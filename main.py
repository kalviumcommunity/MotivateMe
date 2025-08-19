import os
import requests
import json
import re
import argparse
import sys
from typing import Optional, Tuple

# ----------------------
# Local function(s) AI can call
# ----------------------
def get_motivation_by_mood(mood: str) -> dict:
    return {
        "mood": mood,
        "quote": "Keep pushing, greatness takes time.",
        "author": "Anonymous",
        "suggested_action": "Write down one thing you're grateful for today."
    }

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
# High-level function with function calling
# ----------------------
def get_structured_or_function_call(user_mood: str,
                                    api_key: str,
                                    temperature: float = 0.2,
                                    top_k: Optional[int] = None,
                                    top_p: Optional[float] = None) -> Optional[dict]:

    stop_marker = "<END_JSON>"
    system_instr = f"""
    You are a motivational assistant. Generate a motivational quote based on the user's mood.

Return a JSON object with keys: mood, quote, author, suggested_action.
- mood: should match or relate to the user's input mood
- quote: an inspiring motivational quote
- author: the author of the quote
- suggested_action: a practical action the user can take

Respond ONLY with valid JSON, no extra text or formatting. End with {stop_marker}.
"""

    user_line = f'User mood: "{user_mood}".'

    prompt = f"{system_instr}\n{user_line}\nOutput JSON now:\n"

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

    # Validate direct structured output
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
    ap = argparse.ArgumentParser(description="Daily Motivation Bot - Interactive Mode")
    ap.add_argument("--mood", type=str, default=None, help="User mood string (optional)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--api_key", type=str, default=None, help="Google API key (or set GOOGLE_API_KEY env var)")
    args = ap.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("🤖 Welcome to MotivateMe - Your Daily Motivation Bot!")
        print("=" * 50)
        api_key = input("Please enter your Google API key: ").strip()
        if not api_key:
            print("❌ Error: API key is required to run the bot.")
            sys.exit(1)
        print("✅ API key received!")
        print()

    # Get user mood
    user_mood = args.mood
    if not user_mood:
        print("💭 How are you feeling today?")
        user_mood = input("Your mood: ").strip()
        if not user_mood:
            print("❌ Error: Please provide your mood.")
            sys.exit(1)
        print()

    print(f"🎯 Generating motivation for: '{user_mood}'")
    print("⏳ Please wait...")
    print()

    parsed = get_structured_or_function_call(
        user_mood=user_mood,
        api_key=api_key,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    if parsed:
        print("\n" + "=" * 50)
        print("🌟 YOUR MOTIVATION FOR TODAY 🌟")
        print("=" * 50)
        print(f"💭 Mood: {parsed['mood']}")
        print(f"💬 Quote: \"{parsed['quote']}\"")
        print(f"👤 Author: {parsed['author']}")
        print(f"🎯 Suggested Action: {parsed['suggested_action']}")
        print("=" * 50)
        print("💪 Keep going, you've got this!")
    else:
        print("\n❌ [Failed to get valid output]")
        print("Please try again or check your API key.")

if __name__ == "__main__":
    main_cli()
