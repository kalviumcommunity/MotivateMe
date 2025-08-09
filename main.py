# main.py
import os
import requests
import json
from token_utils import log_tokens_after_call  # assumes you created token_utils.py

def call_google_studio(prompt_text: str, api_key: str = None, temperature: float = 0.0, top_k: int | None = None, top_p: float | None = None):
    """
    Calls Google/Generative API (Gemini) with generationConfig including temperature/top_k/top_p.
    Returns (api_response_dict, response_text_str).
    """
    api_key = api_key or os.environ.get("AIzaSyDLjM6tYL5F84PTHcxqgOf37da1T6MQZF0")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY environment variable or pass api_key.")

    # endpoint (Vertex/Gemini) - keep key out of source; we use Authorization if possible
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    generation_config = {"temperature": float(temperature)}
    if top_k is not None:
        generation_config["top_k"] = int(top_k)
    if top_p is not None:
        generation_config["top_p"] = float(top_p)

    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": generation_config
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    api_response = resp.json()

    # Try to extract text (Gemini-like response structure)
    try:
        response_text = api_response["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        response_text = json.dumps(api_response)

    # Log tokens (will print API usage if present, else estimate)
    log_tokens_after_call(api_response=api_response, prompt_text=prompt_text, response_text=response_text, model_name="gemini-1.5-flash")

    return api_response, response_text

if __name__ == "__main__":
    prompt = "Provide a motivational quote for mood: 'tired'. Respond in JSON with fields mood, quote, author, suggested_action."
    # Example: pass top_k=5 to consider the top 5 token candidates
    api_resp, model_text = call_google_studio(prompt_text=prompt, temperature=0.2, top_k=16)
    print("Model Output:", model_text)
