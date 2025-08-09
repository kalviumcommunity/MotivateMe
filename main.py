# main.py
import os
import requests
import json
from token_utils import log_tokens_after_call  # your existing helper

def call_google_studio(prompt_text: str,
                       api_key: str = None,
                       temperature: float = 0.0,
                       top_k: int | None = None,
                       top_p: float | None = None,
                       stop_sequences: list | None = None):
    """
    Calls Google/Generative API (Gemini) with optional stop_sequences.
    Returns (api_response_dict, response_text_str).
    """
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY environment variable or pass api_key.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
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

    # Try to extract the main candidate text (Gemini-like)
    try:
        response_text = api_response["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        response_text = json.dumps(api_response)

    # Log token usage if available or estimate
    log_tokens_after_call(api_response=api_response, prompt_text=prompt_text, response_text=response_text, model_name="gemini-1.5-flash")

    return api_response, response_text


if __name__ == "__main__":
    # Example: we append a clear stop marker in the prompt and pass same marker in stop_sequences
    stop_marker = "<END_OF_RESPONSE>"
    prompt = (
        "You are a supportive motivational coach. "
        "Respond ONLY with a single JSON object with keys: mood, quote, author, suggested_action. "
        "End the response by adding the special token " + stop_marker + " on a new line."
    )

    # call with stop sequence (do NOT hardcode API key here)
    api_resp, model_text = call_google_studio(
        prompt_text=prompt,
        temperature=0.2,
        top_k=16,
        top_p=0.9,
        stop_sequences=["stop_marker"],
        api_key = "AIzaSyDLjM6tYL5F84PTHcxqgOf37da1T6MQZF0"
    )

    # Print the model output (should stop at stop_marker and stop_marker will not be included)
    print("Model Output:", model_text)
