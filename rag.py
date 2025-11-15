import os
import json
from typing import List, Dict, Any, Tuple

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

INDEX_DIR = 'rag_index'
EMBED_MODEL_NAME = os.environ.get('RAG_EMBED_MODEL', 'all-MiniLM-L6-v2')


class RAGStore:
    """Simple RAG store using sentence-transformers + FAISS.

    Provides:
      - init(index_dir)
      - add_documents(list of dicts with id, kiosk, text, meta)
      - search(text, top_k=3, score_threshold=0.3)
      - save(), load()
    """

    def __init__(self, index_dir: str = INDEX_DIR, embed_model_name: str = EMBED_MODEL_NAME):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.meta_path = os.path.join(self.index_dir, 'meta.json')
        self.index_path = os.path.join(self.index_dir, 'index.faiss')
        self.model_name = embed_model_name

        if SentenceTransformer is None:
            raise ImportError('sentence-transformers is required for RAGStore. Install with `pip install sentence-transformers faiss-cpu`')

        self.model = SentenceTransformer(self.model_name)
        self.embeddings = []  # list of vectors
        self.metadatas = []  # list of dicts per vector
        self.index = None
        self._load()

    def _build_index(self):
        import numpy as np
        d = self.embeddings[0].shape[0] if self.embeddings else self.model.get_sentence_embedding_dimension()
        xb = np.vstack(self.embeddings).astype('float32') if self.embeddings else np.zeros((0, d), dtype='float32')
        if faiss is None:
            raise ImportError('faiss is required for RAGStore. Install with `pip install faiss-cpu`')
        # Use IndexFlatIP (inner product) with normalized vectors for cosine similarity
        index = faiss.IndexFlatIP(d)
        # Normalize
        faiss.normalize_L2(xb)
        if xb.shape[0] > 0:
            index.add(xb)
        self.index = index

    def _load(self):
        # load meta + embeddings if present
        try:
            if os.path.exists(self.meta_path):
                with open(self.meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    self.metadatas = meta.get('metadatas', [])
                    # embeddings are not stored as raw floats to save space; recompute on load
                    texts = [m.get('text', '') for m in self.metadatas]
                    if texts:
                        self.embeddings = [self.model.encode(t) for t in texts]
                    else:
                        self.embeddings = []
            else:
                self.metadatas = []
                self.embeddings = []
        except Exception:
            self.metadatas = []
            self.embeddings = []
        # Build FAISS index
        self._build_index()

    def save(self):
        # Save metadata (including original text) and let embeddings be re-computed on load
        meta = {'metadatas': self.metadatas, 'model_name': self.model_name}
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        # Save faiss index separately for faster startup
        if self.index is not None and faiss is not None:
            try:
                faiss.write_index(self.index, self.index_path)
            except Exception:
                # fallback: skip writing
                pass

    def add_documents(self, docs: List[Dict[str, Any]]):
        """Add documents. Each doc should have: id, kiosk, text, meta(optional dict)"""
        import numpy as np
        texts = [d.get('text', '') for d in docs]
        if not texts:
            return
        vecs = self.model.encode(texts)
        # store
        for i, d in enumerate(docs):
            meta = {
                'id': d.get('id'),
                'kiosk': d.get('kiosk'),
                'text': d.get('text'),
                'meta': d.get('meta', {})
            }
            self.metadatas.append(meta)
            self.embeddings.append(vecs[i])

        # rebuild index incrementally: faster approach is to recreate
        self._build_index()
        # add all vectors to FAISS
        if self.index is not None and len(self.embeddings) > 0:
            xb = np.vstack(self.embeddings).astype('float32')
            faiss.normalize_L2(xb)
            # reinitialize index and add
            d = xb.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(xb)

        # persist meta and index
        self.save()

    def search(self, query: str, top_k: int = 3, score_threshold: float = 0.3) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents. Returns list of (meta, score) sorted by score desc."""
        import numpy as np
        if not self.metadatas:
            return []
        qv = self.model.encode([query])[0].astype('float32')
        faiss.normalize_L2(qv.reshape(1, -1))
        if self.index is None:
            self._build_index()
        D, I = self.index.search(qv.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            # cosine similarity in [-1,1]
            s = float(score)
            if s >= score_threshold:
                results.append((self.metadatas[idx], s))
        # sort
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def _demo():
    # quick demo for manual testing
    store = RAGStore()
    store.add_documents([
        {'id': '1', 'kiosk': 'Kiosk 1', 'text': 'Face in image looks synthetic, odd lighting', 'meta': {'trial': 1}},
        {'id': '2', 'kiosk': 'Kiosk 2', 'text': 'QR code with unusual link', 'meta': {'trial': 2}},
    ])
    res = store.search('synthetic face in kiosk image')
    print(res)


if __name__ == '__main__':
    _demo()
