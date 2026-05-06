"""
topic_detector.py
-----------------
Detects topic shifts in the flat chronological message stream using
sliding-window cosine similarity on sentence embeddings.

How it works (step by step):
1.  Messages are in global chronological order (from parser.py)
2.  We embed two consecutive windows of W messages each
3.  cosine_similarity(window_A, window_B) → float in [-1, 1]
4.  If similarity < THRESHOLD → conversation shifted topic → boundary
5.  MIN_TOPIC_LEN prevents micro-topics from single noisy messages
6.  Each topic segment → extractive summary (top TF-IDF sentences)

Output schema per topic checkpoint:
{
    "topic_id":            int,        # 1-indexed
    "start_index":         int,        # global message index (inclusive)
    "end_index":           int,        # global message index (inclusive)
    "message_count":       int,
    "boundary_similarity": float|None, # sim score that triggered the split
    "summary":             str,        # extractive summary of segment
    "key_phrases":         list[str],  # dominant TF-IDF terms for this topic
    "messages":            list[dict]  # full message objects in segment
}

Output schema per 100-message checkpoint (independent of topics):
{
    "checkpoint_id": int,
    "start_index":   int,
    "end_index":     int,
    "message_count": int,
    "summary":       str
}
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from parser import load_messages

try:
    from sentence_transformers import SentenceTransformer
    _HAVE_SBERT = True
except ImportError:
    _HAVE_SBERT = False


# ── Fallback embedder (TF-IDF + LSA) ─────────────────────────────────────────

class _TfidfEmbedder:
    """
    Drop-in replacement for SentenceTransformer when the model
    cannot be downloaded. Uses TF-IDF + TruncatedSVD (LSA) to produce
    dense normalised vectors.
    Fitted once on a corpus sample; subsequent calls use transform().
    """
    def __init__(self, n_components: int = 128):
        self._n = n_components
        self._fitted = False
        self._tfidf = None
        self._svd = None
        self._actual_n = n_components

    def _fit(self, texts: list[str]):
        from sklearn.decomposition import TruncatedSVD
        self._tfidf = TfidfVectorizer(
            stop_words="english", max_features=5000, sublinear_tf=True
        )
        X = self._tfidf.fit_transform(texts)
        n_comp = min(self._n, X.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self._svd.fit(X)
        self._actual_n = n_comp
        self._fitted = True

    def encode(self, texts: list[str], normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        if not self._fitted:
            self._fit(texts)
        X = self._tfidf.transform(texts)
        try:
            vecs = self._svd.transform(X).astype("float32")
        except Exception:
            vecs = X.toarray().astype("float32")[:, :self._actual_n]
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.where(norms == 0, 1, norms)
        return vecs


# ── Constants ─────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = Path(__file__).parent.parent / "data" / "topic_checkpoints.pkl"
MODEL_NAME      = "all-MiniLM-L6-v2"
WINDOW_SIZE     = 8      # messages per comparison window
THRESHOLD       = 0.35   # cosine sim below this → topic boundary
MIN_TOPIC_LEN   = 15     # minimum messages per topic


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_model(messages: list[dict]):
    if _HAVE_SBERT:
        try:
            return SentenceTransformer(MODEL_NAME)
        except Exception:
            pass
    print("  Using TF-IDF+LSA embedder (offline fallback)")
    emb = _TfidfEmbedder()
    emb._fit([m["text"] for m in messages[:8000]])
    return emb


def _window_embedding(model, messages: list[dict], start: int) -> np.ndarray:
    end  = min(start + WINDOW_SIZE, len(messages))
    text = " ".join(m["text"] for m in messages[start:end])
    return model.encode([text], normalize_embeddings=True)


def _extractive_summary(messages: list[dict], n: int = 3) -> str:
    texts = [m["text"] for m in messages]
    if len(texts) <= n:
        return " | ".join(texts)
    try:
        vec   = TfidfVectorizer(stop_words="english", max_features=300)
        mat   = vec.fit_transform(texts).toarray()
        scores = mat.sum(axis=1)
        top   = sorted(np.argsort(scores)[-n:])
        return " ... ".join(texts[i] for i in top)
    except Exception:
        return " | ".join(texts[:n])


def _key_phrases(messages: list[dict], n: int = 6) -> list[str]:
    texts = [m["text"] for m in messages]
    try:
        vec   = TfidfVectorizer(stop_words="english", max_features=500, ngram_range=(1, 2))
        mat   = vec.fit_transform(texts)
        scores = mat.toarray().sum(axis=0)
        terms  = vec.get_feature_names_out()
        top    = np.argsort(scores)[-n:][::-1]
        return [terms[i] for i in top]
    except Exception:
        return []


# ── Core functions ────────────────────────────────────────────────────────────

def detect_topics(messages: list[dict], model=None) -> list[dict]:
    """
    Slide two adjacent windows across the message stream.
    When cosine_similarity(window_i, window_i+1) < THRESHOLD → topic boundary.

    Returns list of topic checkpoint dicts (see module docstring).
    """
    if model is None:
        model = _load_model(messages)

    print(f"[topic_detector] {len(messages):,} messages | "
          f"window={WINDOW_SIZE} | threshold={THRESHOLD} | min_len={MIN_TOPIC_LEN}")

    boundaries     = [0]       # start index of each topic
    boundary_sims  = [None]    # similarity score that triggered the split
    last_boundary  = 0
    step = max(1, WINDOW_SIZE // 2)

    i = 0
    while i + WINDOW_SIZE * 2 <= len(messages):
        if (i - last_boundary) < MIN_TOPIC_LEN:
            i += step
            continue

        emb_a = _window_embedding(model, messages, i)
        emb_b = _window_embedding(model, messages, i + WINDOW_SIZE)
        sim   = float(cosine_similarity(emb_a, emb_b)[0][0])

        if sim < THRESHOLD:
            boundary = i + WINDOW_SIZE
            boundaries.append(boundary)
            boundary_sims.append(round(sim, 4))
            last_boundary = boundary

        i += step

    boundaries.append(len(messages))
    boundary_sims.append(None)

    checkpoints = []
    for t_id, (start, end, sim) in enumerate(
        zip(boundaries, boundaries[1:], boundary_sims[1:]), start=1
    ):
        segment = messages[start:end]
        checkpoints.append({
            "topic_id":            t_id,
            "start_index":         start,
            "end_index":           end - 1,
            "message_count":       len(segment),
            "boundary_similarity": sim,
            "summary":             _extractive_summary(segment, n=3),
            "key_phrases":         _key_phrases(segment, n=6),
            "messages":            segment,
        })

    print(f"[topic_detector] {len(checkpoints)} topics detected")
    return checkpoints


def build_100_msg_checkpoints(messages: list[dict]) -> list[dict]:
    """
    Every 100 messages (independent of topics) → one checkpoint.
    start_index and end_index are global message indices.
    """
    checkpoints = []
    for i in range(0, len(messages), 100):
        segment = messages[i : i + 100]
        checkpoints.append({
            "checkpoint_id": len(checkpoints) + 1,
            "start_index":   i,
            "end_index":     min(i + 99, len(messages) - 1),
            "message_count": len(segment),
            "summary":       _extractive_summary(segment, n=4),
        })
    print(f"[topic_detector] {len(checkpoints)} × 100-msg checkpoints built")
    return checkpoints


def run_and_save(messages: list[dict] = None) -> dict:
    if messages is None:
        messages = load_messages()
    model     = _load_model(messages)
    topic_cps = detect_topics(messages, model=model)
    msg_cps   = build_100_msg_checkpoints(messages)
    data = {
        "topic_checkpoints":   topic_cps,
        "message_checkpoints": msg_cps,
        "total_messages":      len(messages),
        "model_used":          MODEL_NAME if _HAVE_SBERT else "tfidf-lsa",
    }
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"[topic_detector] Saved → {CHECKPOINT_PATH}")
    return data


def load_checkpoints() -> dict:
    with open(CHECKPOINT_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    data = run_and_save()
    print("\nSample topic checkpoints:")
    for cp in data["topic_checkpoints"][:4]:
        print(
            f"  Topic {cp['topic_id']:>3} | "
            f"msgs {cp['start_index']:>6}–{cp['end_index']:<6} | "
            f"sim={cp['boundary_similarity']} | "
            f"phrases={cp['key_phrases'][:3]}"
        )
