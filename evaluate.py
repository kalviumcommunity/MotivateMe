import json
from difflib import SequenceMatcher
import time
import argparse
import random

# === Token logging helper (simple heuristic) ===
def estimate_tokens(text):
    """Rough token estimate: ~4 chars/token."""
    if not text:
        return 0
    return max(1, int(len(text) / 4))

def log_tokens(prompt_text, response_text):
    """Logs estimated token counts after each model call."""
    prompt_tokens = estimate_tokens(prompt_text)
    response_tokens = estimate_tokens(response_text)
    total_tokens = prompt_tokens + response_tokens
    print(f"[Token usage] prompt={prompt_tokens} completion={response_tokens} total={total_tokens}")

# === Files ===
DATASET_FILE = "evaluation_dataset.json"
QUOTES_FILE = "quotes.json"

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def similar(a, b):
    # safe similarity; handle empty strings
    a = (a or "").lower().strip()
    b = (b or "").lower().strip()
    return SequenceMatcher(None, a, b).ratio()

# === Mock model that respects 'temperature' ===
def mock_model(user_input, quotes_db, temperature=0.0):
    """
    Behavior summary:
    - temperature == 0.0: deterministic behavior: return first exact-match quote if present, else fallback
    - 0.0 < temperature < 0.7: mild randomness among matching quotes (if any)
    - temperature >= 0.7: higher chance to return a creative / different quote (random from DB)
    This simulates how temperature affects creativity / randomness in real LLMs.
    """
    user_input_lower = (user_input or "").lower()
    matches = [q for q in quotes_db if q.get("mood", "").lower() in user_input_lower]

    fallback = {
        "mood": user_input,
        "quote": "Keep going, you're doing better than you think.",
        "author": "AI Coach",
        "suggested_action": "Pause and breathe deeply."
    }

    # deterministic
    if temperature <= 0.0:
        if matches:
            return matches[0]
        return fallback

    # mild randomness: choose among matches if they exist
    if 0.0 < temperature < 0.7:
        if matches:
            return random.choice(matches)
        # if no matches, return fallback
        return fallback

    # high temperature: more creative/random - maybe return unrelated quote
    if temperature >= 0.7:
        # with probability = temperature, pick any random quote from DB (creative)
        if quotes_db:
            if random.random() < temperature:
                return random.choice(quotes_db)
        # otherwise, pick from matches or fallback
        if matches:
            return random.choice(matches)
        return fallback

def auto_judge(expected, actual):
    scores = {
        "mood_score": similar(expected.get("mood",""), actual.get("mood","")),
        "quote_score": similar(expected.get("quote",""), actual.get("quote","")),
        "author_score": similar(expected.get("author",""), actual.get("author","")),
        "action_score": similar(expected.get("suggested_action",""), actual.get("suggested_action",""))
    }
    # Weighted overall score: quotes are the most important
    overall = (
        0.25 * scores["mood_score"] +
        0.45 * scores["quote_score"] +
        0.15 * scores["author_score"] +
        0.15 * scores["action_score"]
    )
    scores["overall_score"] = overall
    return scores

def run_evaluation(temperature=0.0):
    dataset = load_json(DATASET_FILE)
    quotes_db = load_json(QUOTES_FILE)
    total_time = 0
    results = []

    print(f"\n[INFO] Running evaluation with temperature={temperature}\n")

    for sample in dataset:
        start = time.perf_counter()

        # Call the mock model with temperature
        actual = mock_model(sample["input"], quotes_db, temperature=temperature)

        elapsed = time.perf_counter() - start
        total_time += elapsed

        # Log tokens for this call
        log_tokens(sample["input"], json.dumps(actual))

        # Also print the temperature used for this sample
        print(f"[Model call] temperature={temperature:.2f} input={sample['input']}")

        scores = auto_judge(sample["expected"], actual)
        results.append({
            "id": sample.get("id"),
            "input": sample["input"],
            "expected": sample["expected"],
            "actual": actual,
            "scores": scores,
            "latency_s": elapsed,
            "temperature": temperature
        })

        print(f" -> sample {sample.get('id')} overall_score={scores['overall_score']:.3f} time={elapsed:.3f}s\n")

    avg_score = sum(r["scores"]["overall_score"] for r in results) / len(results)
    avg_latency = total_time / len(results)

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Samples: {len(results)}")
    print(f"Average overall score: {avg_score:.3f}")
    print(f"Average latency (s): {avg_latency:.3f}")
    print("Results retained in memory (and printed above).")
    return {"summary": {"average_overall_score": avg_score, "average_latency_s": avg_latency, "num_samples": len(results)}, "results": results}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run evaluation with optional temperature")
    ap.add_argument("--temperature", type=float, default=0.0, help="Temperature value to use for model calls (0.0 deterministic → higher more random)")
    args = ap.parse_args()
    run_evaluation(temperature=args.temperature)
