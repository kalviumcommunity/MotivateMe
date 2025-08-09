import json
from difflib import SequenceMatcher
import time

# === Token logging helper ===
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

# === Existing code ===
DATASET_FILE = "evaluation_dataset.json"
QUOTES_FILE = "quotes.json"

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def similar(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def mock_model(user_input, quotes_db):
    """Basic mock: pick first quote whose mood appears in input."""
    user_input_lower = user_input.lower()
    for q in quotes_db:
        if q["mood"].lower() in user_input_lower:
            return q
    return {
        "mood": "unknown",
        "quote": "Keep going, you're doing better than you think.",
        "author": "AI Coach",
        "suggested_action": "Pause and breathe deeply."
    }

def auto_judge(expected, actual):
    scores = {
        "mood_score": similar(expected["mood"], actual["mood"]),
        "quote_score": similar(expected["quote"], actual["quote"]),
        "author_score": similar(expected["author"], actual["author"]),
        "action_score": similar(expected["suggested_action"], actual["suggested_action"])
    }
    # Weighted overall score
    overall = (
        0.25 * scores["mood_score"] +
        0.45 * scores["quote_score"] +
        0.15 * scores["author_score"] +
        0.15 * scores["action_score"]
    )
    scores["overall_score"] = overall
    return scores

def run_evaluation():
    dataset = load_json(DATASET_FILE)
    quotes_db = load_json(QUOTES_FILE)
    total_time = 0
    results = []

    for sample in dataset:
        start = time.perf_counter()
        actual = mock_model(sample["input"], quotes_db)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        # Log tokens for this call
        log_tokens(sample["input"], json.dumps(actual))

        scores = auto_judge(sample["expected"], actual)
        results.append({
            "input": sample["input"],
            "expected": sample["expected"],
            "actual": actual,
            "scores": scores,
            "latency_s": elapsed
        })

    avg_score = sum(r["scores"]["overall_score"] for r in results) / len(results)
    avg_latency = total_time / len(results)

    print("\n=== Evaluation Summary ===")
    print(f"Average Score: {avg_score:.3f}")
    print(f"Average Latency: {avg_latency:.3f} seconds\n")

    for r in results:
        print(f"Input: {r['input']}")
        print(f"Score: {r['scores']['overall_score']:.3f}")
        print()

if __name__ == "__main__":
    run_evaluation()
