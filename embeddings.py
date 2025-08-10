# embeddings.py
import os
import json
import requests
from typing import List, Optional

# Optional local embedding model (sentence-transformers)
def embed_texts_local(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Compute embeddings locally using sentence-transformers.
    Returns list of float vectors.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        vecs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in vecs]
    except Exception as e:
        raise RuntimeError("Local embedding failed. Install sentence-transformers or use provider mode. " + str(e))

def embed_texts_provider(texts: List[str],
                         api_url: str,
                         api_key: Optional[str] = None,
                         request_kwargs: Optional[dict] = None) -> List[List[float]]:
    """
    Generic provider call. You MUST pass api_url for your provider's embeddings endpoint.
    The function assumes a simple request/response shape but includes flexible parsing.
    Example provider usage (Google/Azure/OpenAI may have different shapes):
      payload = {"input": texts}
      response: { "embeddings": [ { "embedding": [...] }, ... ] }
    """
    headers = {"Content-Type": "application/json"}
    payload = {"input": texts}
    if request_kwargs:
        payload.update(request_kwargs)

    # Allow API key in query if present
    url = api_url
    if api_key:
        # if provider expects key in URL parameter
        if "?" in api_url:
            url = f"{api_url}&key={api_key}"
        else:
            url = f"{api_url}?key={api_key}"

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    resp = r.json()

    # Try common response shapes
    if isinstance(resp, dict):
        # 1) shape: {"embeddings": [{ "embedding": [...] }, ...]}
        if "embeddings" in resp:
            out = []
            for item in resp["embeddings"]:
                if isinstance(item, dict) and "embedding" in item:
                    out.append(item["embedding"])
                elif isinstance(item, list):
                    out.append(item)
                else:
                    # unknown format - append raw
                    out.append(item)
            return out

        # 2) openai-like: {"data":[ {"embedding": [...]}, ... ]}
        if "data" in resp and isinstance(resp["data"], list):
            out = []
            for item in resp["data"]:
                if isinstance(item, dict) and "embedding" in item:
                    out.append(item["embedding"])
            if out:
                return out

        # 3) google generative style may return: top-level candidates (less common for embeddings)
        # fallthrough: try to locate any nested embedding lists
        def find_embeddings_in_obj(o):
            if isinstance(o, list):
                # check if inner lists look like floats
                if o and isinstance(o[0], (list, float, int)):
                    # if it's list of floats -> treat as one embedding or list of embeddings
                    if all(isinstance(x, (float, int)) for x in o):
                        return [o]
                    # if list of lists -> treat as multiple embeddings
                    if all(isinstance(x, list) for x in o):
                        return o
                # otherwise search children
                for x in o:
                    res = find_embeddings_in_obj(x)
                    if res:
                        return res
            elif isinstance(o, dict):
                for v in o.values():
                    res = find_embeddings_in_obj(v)
                    if res:
                        return res
            return None

        guessed = find_embeddings_in_obj(resp)
        if guessed:
            return guessed

    raise RuntimeError("Cannot parse embeddings from provider response: " + json.dumps(resp)[:1000])

# Utility: save embeddings as jsonl
def save_embeddings_jsonl(texts: List[str], embeddings: List[List[float]], out_path: str = "embeddings.jsonl"):
    assert len(texts) == len(embeddings)
    with open(out_path, "w", encoding="utf-8") as f:
        for t, e in zip(texts, embeddings):
            json.dump({"text": t, "embedding": e}, f, ensure_ascii=False)
            f.write("\n")
    print(f"[Saved] {len(embeddings)} embeddings -> {out_path}")

# Utility: simple cosine similarity search (in-memory, numpy required)
def cosine_similarity(a, b):
    import math
    # a,b are lists
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na*nb)

def simple_search(query_emb, embeddings, top_k=3):
    scores = [(i, cosine_similarity(query_emb, emb)) for i, emb in enumerate(embeddings)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
