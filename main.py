# main.py
import requests
import json
from token_utils import log_tokens_after_call  # assumes you have this from your token logging task

def call_google_studio(prompt_text: str, api_key: str, temperature: float = 0.0):
    """
    Calls Google Studio API with a prompt and temperature.
    Returns: (api_response_dict_or_none, response_text_str)
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={AIzaSyDLjM6tYL5F84PTHcxqgOf37da1T6MQZF0}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": temperature
        }
    }

    r = requests.post(url, headers=headers, json=payload, timeout=20)
    api_response = r.json()

    # Try extracting the main model output text
    try:
        response_text = api_response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        response_text = json.dumps(api_response)

    return api_response, response_text


if __name__ == "__main__":
    API_KEY = "AIzaSyDLjM6tYL5F84PTHcxqgOf37da1T6MQZF0"  # replace with env var or secure store
    prompt = "Provide a motivational quote for mood: 'tired'. Respond in JSON with fields mood, quote, author, suggested_action."

    # Example call with temperature control
    api_resp, model_text = call_google_studio(prompt, api_key=API_KEY, temperature=0.7)

    # Log token usage
    log_tokens_after_call(
        api_response=api_resp,
        prompt_text=prompt,
        response_text=model_text,
        model_name="gemini-1.5-flash"
    )

    # Print the model's output
    print("Model Output:", model_text)
