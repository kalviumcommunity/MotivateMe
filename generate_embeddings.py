# generate_embeddings.py
import os
import argparse
import json
from embeddings import embed_texts_local, embed_texts_provider, save_embeddings_jsonl

def load_quotes(path="quotes.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["local", "provider"], default="local",
                    help="local uses sentence-transformers; provider uses remote endpoint")
    ap.add_argument("--api_url", type=str, default=None, help="Provider embeddings URL (if provider mode)")
    ap.add_argument("--api_key", type=str, default=None, help="Provider API key (if provider mode)")
    ap.add_argument("--model", type=str, default=None, help="Local model name (optional)")
    ap.add_argument("--out", type=str, default="embeddings.jsonl")
    args = ap.parse_args()

    quotes = load_quotes()
    # We'll embed the quote text (prefer 'quote' key if present)
    texts = []
    for item in quotes:
        # item may be dict with quote string
        if isinstance(item, dict) and "quote" in item:
            texts.append(item["quote"])
        elif isinstance(item, str):
            texts.append(item)
        else:
            texts.append(str(item))

    if args.provider == "local":
        print("[Mode] local embedding using sentence-transformers")
        embeddings = embed_texts_local(texts, model_name=args.model or "all-MiniLM-L6-v2")
    else:
        if not args.api_url:
            raise SystemExit("Provider mode requires --api_url")
        print("[Mode] provider embedding:", args.api_url)
        embeddings = embed_texts_provider(texts, api_url=args.api_url, api_key=args.api_key)

    save_embeddings_jsonl(texts, embeddings, out_path=args.out)
    print("Done.")

if __name__ == "__main__":
    main()

