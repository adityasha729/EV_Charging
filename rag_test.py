"""
Requirements:
pip install -U sentence-transformers faiss-cpu numpy
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os


MODEL_NAME = "all-mpnet-base-v2"   
EMBED_DIM = 768                    
INDEX_PATH = "kiosk_faiss.index"
META_PATH = "kiosk_metadata.json"

# Use cosine similarity via inner-product on L2-normalized vectors
index = faiss.IndexFlatIP(EMBED_DIM)  # inner product index
model = SentenceTransformer(MODEL_NAME)

metadata = []  

# ---------- Helpers ----------
def embed_texts(texts):
    """
    Returns L2-normalized embeddings (float32) for a list of texts.
    """
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # ensure float32
    embs = embs.astype("float32")
    # L2 normalize for cosine similarity with IndexFlatIP
    faiss.normalize_L2(embs)
    return embs

def store_description(text, cyberattack, extra_meta=None):
    """
    Stores the description in FAISS and the metadata list.
    """
    if extra_meta is None:
        extra_meta = {}
    embs = embed_texts([text])  # shape (1, EMBED_DIM)
    index.add(embs)
    metadata.append({
        "text": text,
        "cyberattack": bool(cyberattack),
        "meta": extra_meta
    })

def retrieve_similar(text, top_k=3):
    """
    Returns top_k similar stored descriptions (with similarity scores).
    """
    if index.ntotal == 0:
        return []

    q_emb = embed_texts([text])
    # perform search
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx]
        results.append({
            "text": item["text"],
            "cyberattack": item["cyberattack"],
            "meta": item["meta"],
            "score": float(score)   
        })
    return results


def save_state(index_path=INDEX_PATH, meta_path=META_PATH):
    # save faiss index
    faiss.write_index(index, index_path)
    # save metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_state(index_path=INDEX_PATH, meta_path=META_PATH):
    global index, metadata
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        # rebuild blank index if not found
        index = faiss.IndexFlatIP(EMBED_DIM)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = []


if __name__ == "__main__":
    # start fresh or load existing
    load_state()

    # store some examples
    store_description("The kiosk displayed a 70% coupon with a distorted celebrity face.", True, {"station": 5, "kiosk": 2})
    store_description("Normal charging UI with no suspicious offers.", False, {"station": 2, "kiosk": 1})
    store_description("Free QR code for discount; required email login form.", True, {"station": 8, "kiosk": 3})

    # query
    query = "A flashy 60% discount coupon shows a celebrity photo and asks for email login."
    results = retrieve_similar(query, top_k=3)

    print("Query:", query)
    print("\nTop matches:")
    for r in results:
        print(f"- score={r['score']:.4f} attack={r['cyberattack']} meta={r['meta']}")
        print(f"  text: {r['text']}\n")

    # persist
    save_state()
