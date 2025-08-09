# evaluate.py
import json
from difflib import SequenceMatcher
import time
import argparse
import random

# === Token logging helper (simple heuristic) ===
def estimate_tokens(text):
    if not text:
        return 0
    return max(1, int(len(text) / 4))

def log_tokens(prompt_text, response_text):
    prompt_tokens = estimate_tokens(prompt_text)
    response_tokens = estimate_tokens(response_text)
    total_tokens = prompt_tokens + response_tokens
    print(f"[Token usage] prompt={prompt_tokens} completion={response_tokens} total={total_tokens}")

DATASET_FILE = "evaluation_dataset.json"
QUOTES_FILE = "quotes.json"

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def similar(a, b):
    a = (a or "").lower().strip()
    b = (b or "").lower().strip()
    return SequenceMatcher(None, a, b).ratio()

def mock_model(user_input, quotes_db, temperature=0.0, top_k: int | None = None):
    """
    Mock behavior demonstrating temperature + top_k:
    - top_k limits how many candidate quotes are considered.
    - temperature controls randomness among those candidates.
    """
    user_input_lower = (user_input or "").lower()
    # find matches where the mood substring appears
    matches = [q for q in quotes_db if q.get("mood", "").lower() in user_input_lower]

    # If top_k provided, limit matching pool size:
    if top_k is not None and top_k > 0:
        if matches:
            # If matches >= top_k, random sample top_k of them; else use all matches
            pool = matches[:top_k] if len(matches) <= top_k else random.sample(matches, k=top_k)
        else:
            # No direct matches: consider top_k random quotes as broader candidates
            pool = random.sample(quotes_db, k=min(top_k, len(quotes_db)))
    else:
        # no top_k applied: use matches or entire DB fallback
        pool = matches if matches else quotes_db

    # selection strategy:
    if temperature <= 0.0:
        # deterministic: pick first item in pool
        chosen = pool[0] if pool else {
            "mood": user_input, "quote": "Keep going, you're doing better than you think.",
            "author": "AI Coach", "suggested_action": "Pause and breathe."
        }
        return chosen

    # temperature > 0: pick random from pool, bias by temperature (we keep simple)
    # higher temperature => more random selection over pool
    if pool:
        # with higher temperature, weigh randomness heavier; we just random.choice for demo
        return random.choice(pool)
    return {"mood": user_input, "quote": "Keep going, you're doing better than you think.", "author": "AI Coach", "suggested_action": "Pause and breathe."}

def auto_judge(expected, actual):
    scores = {
        "mood_score": similar(expected.get("mood",""), actual.get("mood","")),
        "quote_score": similar(expected.get("quote",""), actual.get("quote","")),
        "author_score": similar(expected.get("author",""), actual.get("author","")),
        "action_score": similar(expected.get("suggested_action",""), actual.get("suggested_action",""))
    }
    overall = (
        0.25 * scores["mood_score"] +
        0.45 * scores["quote_score"] +
        0.15 * scores["author_score"] +
        0.15 * scores["action_score"]
    )
    scores["overall_score"] = overall
    return scores

def run_evaluation(temperature=0.0, top_k=None):
    dataset = load_json(DATASET_FILE)
    quotes_db = load_json(QUOTES_FILE)
    total_time = 0
    results = []

    print(f"\n[INFO] Running evaluation with temperature={temperature} top_k={top_k}\n")

    for sample in dataset:
        start = time.perf_counter()
        actual = mock_model(sample["input"], quotes_db, temperature=temperature, top_k=top_k)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        # token logging
        log_tokens(sample["input"], json.dumps(actual))

        print(f"[Model call] temperature={temperature:.2f} top_k={top_k} input={sample['input']}")
        scores = auto_judge(sample["expected"], actual)
        results.append({
            "id": sample.get("id"),
            "input": sample["input"],
            "expected": sample["expected"],
            "actual": actual,
            "scores": scores,
            "latency_s": elapsed,
            "temperature": temperature,
            "top_k": top_k
        })

        print(f" -> sample {sample.get('id')} overall_score={scores['overall_score']:.3f} time={elapsed:.3f}s\n")

    avg_score = sum(r["scores"]["overall_score"] for r in results) / len(results)
    avg_latency = total_time / len(results)
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Samples: {len(results)}")
    print(f"Average overall score: {avg_score:.3f}")
    print(f"Average latency (s): {avg_latency:.3f}")
    return {"summary": {"average_overall_score": avg_score, "average_latency_s": avg_latency, "num_samples": len(results)}, "results": results}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=None, help="Limit candidate pool size (Top-K)")
    args = ap.parse_args()
    run_evaluation(temperature=args.temperature, top_k=args.top_k)
