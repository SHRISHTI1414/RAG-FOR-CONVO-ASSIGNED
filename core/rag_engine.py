"""
rag_engine.py
-------------
Builds a FAISS vector index over:
  1. Topic checkpoint summaries  (semantic overview of a topic segment)
  2. 100-message checkpoint summaries  (coarser, time-based)
  3. Overlapping 25-message raw chunks  (fine-grained message context)

Retrieval at query time:
  - Embed the query with the same model used to build the index
  - Run FAISS inner-product search (≡ cosine sim on unit vectors)
  - Return ranked results split by source type
  - Combine into a structured context + citation dict for the generator

Why FAISS IndexFlatIP:
  - Exact (not approximate) search
  - Inner product on L2-normalised vectors = cosine similarity
  - No external server, ~10ms for 15k vectors on CPU
"""

import pickle
import numpy as np
import faiss
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    _HAVE_SBERT = True
except ImportError:
    _HAVE_SBERT = False

from parser import load_messages
from topic_detector import load_checkpoints, run_and_save as build_checkpoints, _TfidfEmbedder

MODEL_NAME  = "all-MiniLM-L6-v2"
INDEX_PATH  = Path(__file__).parent.parent / "data" / "faiss_index.pkl"
CHUNK_SIZE  = 25   # messages per retrieval chunk
CHUNK_STEP  = 12   # overlap step (50 % overlap)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunks(messages: list[dict]) -> list[dict]:
    """
    Split the flat message list into overlapping windows of CHUNK_SIZE.
    Overlap prevents relevant context from being split across chunk boundaries.
    """
    chunks = []
    for i in range(0, len(messages), CHUNK_STEP):
        seg = messages[i : i + CHUNK_SIZE]
        if len(seg) < 5:
            continue
        chunks.append({
            "chunk_id":  len(chunks),
            "start_idx": seg[0]["global_idx"],
            "end_idx":   seg[-1]["global_idx"],
            "text":      " ".join(m["text"] for m in seg),
            "source":    "message_chunk",
        })
    return chunks


def _load_or_fallback_model(messages: list[dict] = None):
    if _HAVE_SBERT:
        try:
            return SentenceTransformer(MODEL_NAME)
        except Exception:
            pass
    emb = _TfidfEmbedder()
    if messages:
        emb._fit([m["text"] for m in messages[:8000]])
    return emb


# ── Build ─────────────────────────────────────────────────────────────────────

def build_index(messages: list[dict] = None, checkpoints: dict = None) -> dict:
    """
    Embed all documents and build a FAISS IndexFlatIP.
    Saves index + metadata + fitted model to disk.
    """
    if messages is None:
        messages = load_messages()
    if checkpoints is None:
        try:
            checkpoints = load_checkpoints()
        except FileNotFoundError:
            checkpoints = build_checkpoints(messages)

    model = _load_or_fallback_model(messages)

    # ── Build document list ───────────────────────────────────────────────────
    docs = []

    for cp in checkpoints["topic_checkpoints"]:
        docs.append({
            "text":   cp["summary"],
            "source": "topic_summary",
            "meta": {
                "topic_id":    cp["topic_id"],
                "start_index": cp["start_index"],
                "end_index":   cp["end_index"],
                "message_count": cp["message_count"],
                "key_phrases": cp.get("key_phrases", []),
                "boundary_similarity": cp.get("boundary_similarity"),
            },
        })

    for cp in checkpoints["message_checkpoints"]:
        docs.append({
            "text":   cp["summary"],
            "source": "100msg_checkpoint",
            "meta": {
                "checkpoint_id": cp["checkpoint_id"],
                "start_index":   cp["start_index"],
                "end_index":     cp["end_index"],
            },
        })

    for chunk in _make_chunks(messages):
        docs.append({
            "text":   chunk["text"],
            "source": "message_chunk",
            "meta": {
                "start_idx": chunk["start_idx"],
                "end_idx":   chunk["end_idx"],
            },
        })

    print(f"[rag_engine] Indexing {len(docs):,} documents")
    print(f"  topic_summary:     {len(checkpoints['topic_checkpoints'])}")
    print(f"  100msg_checkpoint: {len(checkpoints['message_checkpoints'])}")
    print(f"  message_chunk:     {len(docs) - len(checkpoints['topic_checkpoints']) - len(checkpoints['message_checkpoints'])}")

    # ── Embed ─────────────────────────────────────────────────────────────────
    texts = [d["text"] for d in docs]
    batch = 256
    embs  = []
    for i in range(0, len(texts), batch):
        embs.append(
            model.encode(texts[i : i + batch], normalize_embeddings=True, show_progress_bar=False)
        )
        if i % 5000 == 0:
            print(f"  embedded {i}/{len(texts)}...")
    embeddings = np.vstack(embs).astype("float32")

    # ── FAISS index ───────────────────────────────────────────────────────────
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)    # inner product = cosine sim (vecs are L2-normalised)
    index.add(embeddings)

    print(f"[rag_engine] FAISS index: {index.ntotal} vectors, dim={dim}")

    payload = {
        "index":       index,
        "docs":        docs,
        "dim":         dim,
        "model_name":  MODEL_NAME,
        "fitted_model": model,   # save so RAGEngine reuses the exact same model
    }
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"[rag_engine] Saved → {INDEX_PATH}")
    return payload


def load_index() -> dict:
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)


# ── Engine ────────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Wraps the FAISS index and handles query embedding + ranked retrieval.

    retrieve() returns a structured dict:
    {
        "query": str,
        "topic_summaries": [
            {
                "rank": 1,
                "score": 0.82,          # cosine similarity
                "source": "topic_summary",
                "meta": { topic_id, start_index, end_index, key_phrases, ... },
                "text": "..."
            },
            ...
        ],
        "message_chunks": [ ... same structure ... ]
    }
    """

    def __init__(self, payload: dict = None):
        if payload is None:
            payload = load_index()
        self.index = payload["index"]
        self.docs  = payload["docs"]

        # Reuse the exact fitted model from build time — avoids re-fitting TF-IDF
        if payload.get("fitted_model") is not None:
            self.model = payload["fitted_model"]
        elif _HAVE_SBERT:
            try:
                self.model = SentenceTransformer(payload["model_name"])
            except Exception:
                self.model = _TfidfEmbedder()
        else:
            self.model = _TfidfEmbedder()

    def retrieve(
        self,
        query: str,
        top_k_topics: int = 3,
        top_k_chunks: int = 3,
    ) -> dict:
        """
        Embed query → FAISS search → split results by source type.
        Results are ranked by cosine similarity score (descending).
        """
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")

        # Fetch extra candidates so we can filter by type
        k         = (top_k_topics + top_k_chunks) * 6
        scores, idxs = self.index.search(q_emb, min(k, self.index.ntotal))

        topic_hits = []
        chunk_hits = []

        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
            if idx == -1:
                continue
            doc = self.docs[idx]
            hit = {
                "rank":   rank,
                "score":  round(float(score), 4),
                "source": doc["source"],
                "meta":   doc["meta"],
                "text":   doc["text"],
            }
            if doc["source"] == "topic_summary" and len(topic_hits) < top_k_topics:
                topic_hits.append(hit)
            elif doc["source"] == "message_chunk" and len(chunk_hits) < top_k_chunks:
                chunk_hits.append(hit)
            if len(topic_hits) >= top_k_topics and len(chunk_hits) >= top_k_chunks:
                break

        return {
            "query":          query,
            "topic_summaries": topic_hits,
            "message_chunks":  chunk_hits,
        }

    def build_context(self, retrieval: dict) -> str:
        """
        Format retrieval results into a context string for the generator.
        Includes similarity scores so the generator (and evaluators) can see
        how confident each retrieval is.
        """
        parts = []

        if retrieval["topic_summaries"]:
            parts.append("=== Relevant Topic Segments (by similarity) ===")
            for hit in retrieval["topic_summaries"]:
                m = hit["meta"]
                parts.append(
                    f"[Topic {m.get('topic_id', '?')} | "
                    f"msgs {m.get('start_index', '?')}–{m.get('end_index', '?')} | "
                    f"score={hit['score']}]\n"
                    f"Key phrases: {', '.join(m.get('key_phrases', [])[:4])}\n"
                    f"{hit['text']}"
                )

        if retrieval["message_chunks"]:
            parts.append("\n=== Relevant Message Excerpts (by similarity) ===")
            for hit in retrieval["message_chunks"]:
                m = hit["meta"]
                parts.append(
                    f"[Messages {m.get('start_idx', '?')}–{m.get('end_idx', '?')} | "
                    f"score={hit['score']}]\n"
                    f"{hit['text'][:500]}..."
                )

        return "\n\n".join(parts)


if __name__ == "__main__":
    messages = load_messages()
    payload  = build_index(messages)
    engine   = RAGEngine(payload)

    q = "What does the user like to do for fun?"
    r = engine.retrieve(q)
    print(f"\nQuery: {q}")
    print(f"Top topic hit  : score={r['topic_summaries'][0]['score']} | "
          f"topic {r['topic_summaries'][0]['meta']['topic_id']}")
    print(f"Top chunk hit  : score={r['message_chunks'][0]['score']}")
    print("\nContext:\n", engine.build_context(r)[:600])
