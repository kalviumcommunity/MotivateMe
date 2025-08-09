# evaluate.py
import json
from difflib import SequenceMatcher
import time
import argparse
import random
import math

# token logging helper (simple heuristic)
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

def mock_model(user_input, quotes_db, temperature=0.0, top_k: int | None = None, top_p: float | None = None):
    """
    Mock behaviour demonstrating temperature, top_k and top_p:
    - top_k limits candidate pool (number)
    - top_p simulates nucleus sampling by taking a prefix of the pool of size proportional to top_p
    - temperature controls randomness within that pool
    """
    user_input_lower = (user_input or "").lower()
    # matches where mood substring appears
    matches = [q for q in quotes_db if q.get("mood", "").lower() in user_input_lower]

    # if matches empty, fallback to full DB as potential candidates
    candidate_pool = matches if matches else quotes_db.copy()

    # apply top_k: shrink candidate_pool to at most top_k random items (if provided)
    if top_k is not None and top_k > 0:
        if len(candidate_pool) > top_k:
            candidate_pool = random.sample(candidate_pool, k=top_k)

    # apply top_p: simulate nucleus by taking a prefix sized by top_p * len(candidate_pool)
    if top_p is not None and 0.0 < top_p < 1.0 and candidate_pool:
        # determine number to keep; ensure at least 1
        keep = max(1, int(math.ceil(len(candidate_pool) * float(top_p))))
        # For deterministic feel, sort candidate_pool by mood (stable), and take first 'keep' items
        candidate_pool = candidate_pool[:keep]

    # selection strategy combining temperature:
    if not candidate_pool:
        return {
            "mood": user_input,
            "quote": "Keep going, you're doing better than you think.",
            "author": "AI Coach",
            "suggested_action": "Pause and breathe."
        }

    if temperature <= 0.0:
        # deterministic: pick first in pool
        return candidate_pool[0]

    # temperature > 0: random pick from candidate_pool
    return random.choice(candidate_pool)

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

def run_evaluation(temperature=0.0, top_k=None, top_p=None):
    dataset = load_json(DATASET_FILE)
    quotes_db = load_json(QUOTES_FILE)
    total_time = 0
    results = []

    print(f"\n[INFO] Running evaluation with temperature={temperature}, top_k={top_k}, top_p={top_p}\n")

    for sample in dataset:
        start = time.perf_counter()
        actual = mock_model(sample["input"], quotes_db, temperature=temperature, top_k=top_k, top_p=top_p)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        # token logging
        log_tokens(sample["input"], json.dumps(actual))

        print(f"[Model call] temp={temperature:.2f} top_k={top_k} top_p={top_p} input={sample['input']}")
        scores = auto_judge(sample["expected"], actual)
        results.append({
            "id": sample.get("id"),
            "input": sample["input"],
            "expected": sample["expected"],
            "actual": actual,
            "scores": scores,
            "latency_s": elapsed,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
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
    ap.add_argument("--top_p", type=float, default=None, help="Nucleus sampling parameter (0.0 < top_p <= 1.0)")
    args = ap.parse_args()
    run_evaluation(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
