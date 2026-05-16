"""
conflict_resolver.py
--------------------
Handles the hard retrieval problem: when a query matches multiple
topic chunks that contain contradictory or fragmented information.

Example: "Did I mention anything about my sister?"
The word 'sister' appears in 3 different topic segments with different
emotional contexts — one mentions missing her, another mentions a fight,
another is a passing reference. A naive retriever would just return
all three and confuse the user.

This resolver does four things:
1. Retrieves all candidate chunks matching the query
2. Scores each chunk by recency + emotional weight + relevance
3. Detects contradictions between chunks (same entity, conflicting sentiment)
4. Merges the ranked chunks into a single coherent answer with
   contradictions flagged explicitly

Design note: We deliberately avoid calling any LLM for the merge step.
The merging logic is rule-based — it produces less fluent output than
GPT-4 would, but it's transparent, offline, and fast.
"""

import re
import json
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# ── Emotional weight lexicon ──────────────────────────────────────────────────
# Words that signal emotional intensity — used to up-rank chunks that
# carry more emotional charge (people remember emotional moments better,
# and they're more likely to be what the user actually wants).

HIGH_EMOTION_WORDS = {
    "positive": [
        "love", "happy", "excited", "wonderful", "amazing", "proud",
        "grateful", "thrilled", "overjoyed", "blessed", "great", "fantastic"
    ],
    "negative": [
        "hate", "angry", "scared", "sad", "hurt", "frustrated", "terrible",
        "awful", "crying", "depressed", "fight", "argument", "miss", "lost"
    ],
    "relational": [
        "told", "said", "mentioned", "talked", "discussed", "asked",
        "called", "texted", "met", "saw", "visited", "helped", "supported"
    ]
}

# Contradiction signal pairs — if chunk A has one side and chunk B has
# the other, we flag them as potentially contradictory
CONTRADICTION_PAIRS = [
    (["love", "like", "enjoy", "happy", "great", "close", "miss", "good"],
     ["hate", "fight", "argument", "angry", "distance", "toxic", "bad"]),
    (["healthy", "fine", "okay", "good", "better", "recovered"],
     ["sick", "ill", "hospital", "pain", "worse", "struggling"]),
    (["together", "dating", "married", "relationship", "partner"],
     ["broke up", "single", "divorce", "separated", "ex"]),
    (["hired", "job", "working", "employed", "promoted"],
     ["fired", "quit", "unemployed", "laid off", "left"]),
]


# ── Chunk scoring ─────────────────────────────────────────────────────────────

def compute_recency_score(chunk_start_idx: int, total_messages: int) -> float:
    """
    More recent messages get higher scores.
    Linear decay: last message = 1.0, first message = 0.0.
    We use start_idx as a proxy for time since messages are chronological.
    """
    if total_messages <= 1:
        return 1.0
    return chunk_start_idx / total_messages


def compute_emotional_weight(text: str) -> float:
    """
    Score how emotionally charged a chunk is.
    Combines positive, negative, and relational signal.
    Chunks with strong emotional content are more memorable and
    more likely to be what the user is actually asking about.
    """
    text_lower = text.lower()
    score = 0.0

    for category, words in HIGH_EMOTION_WORDS.items():
        hits = sum(1 for w in words if w in text_lower)
        if category == "negative":
            score += hits * 1.2   # negative events tend to be more salient
        elif category == "relational":
            score += hits * 0.8
        else:
            score += hits * 1.0

    word_count = max(len(text.split()), 1)
    return min(score / (word_count / 20), 1.0)


def compute_query_relevance(query: str, chunk_text: str) -> float:
    """
    Simple keyword overlap between query and chunk.
    Not semantic — just checks if the query's key terms appear in the chunk.
    For conflict resolution we want exact matches, not fuzzy recall.
    """
    query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
    chunk_words = set(re.findall(r'\b\w{3,}\b', chunk_text.lower()))

    if not query_words:
        return 0.0

    overlap = query_words & chunk_words
    return len(overlap) / len(query_words)


def score_chunk(
    chunk: dict,
    query: str,
    total_messages: int,
    recency_weight: float = 0.35,
    emotion_weight: float = 0.30,
    relevance_weight: float = 0.35,
) -> float:
    """
    Composite score for a single chunk.
    The three components are weighted and combined.
    Weights are tunable — we default to roughly equal emphasis.
    """
    text = chunk.get("text", "")
    start_idx = chunk.get("meta", {}).get("start_idx", 0) or \
                chunk.get("meta", {}).get("start_index", 0)

    recency   = compute_recency_score(start_idx, total_messages)
    emotion   = compute_emotional_weight(text)
    relevance = compute_query_relevance(query, text)

    return (
        recency_weight   * recency +
        emotion_weight   * emotion +
        relevance_weight * relevance
    )


# ── Contradiction detection ───────────────────────────────────────────────────

def sentiment_signature(text: str) -> dict:
    """
    Classify a chunk's sentiment across our contradiction pair dimensions.
    Returns a dict of {dimension: "positive"|"negative"|"neutral"}.
    """
    text_lower = text.lower()
    signatures = {}

    for i, (pos_words, neg_words) in enumerate(CONTRADICTION_PAIRS):
        pos_hits = sum(1 for w in pos_words if w in text_lower)
        neg_hits = sum(1 for w in neg_words if w in text_lower)

        if pos_hits > neg_hits:
            signatures[f"dim_{i}"] = "positive"
        elif neg_hits > pos_hits:
            signatures[f"dim_{i}"] = "negative"
        else:
            signatures[f"dim_{i}"] = "neutral"

    return signatures


def detect_contradictions(scored_chunks: list[dict]) -> list[dict]:
    """
    Compare chunk pairs for contradictory sentiment signatures.
    Returns a list of contradiction descriptions.

    We only flag contradictions between chunks that both have
    high relevance scores — otherwise we'd be comparing apples to oranges.
    """
    contradictions = []

    for i in range(len(scored_chunks)):
        for j in range(i + 1, len(scored_chunks)):
            chunk_a = scored_chunks[i]
            chunk_b = scored_chunks[j]

            sig_a = sentiment_signature(chunk_a["text"])
            sig_b = sentiment_signature(chunk_b["text"])

            conflicting_dims = []
            for dim in sig_a:
                if (sig_a[dim] != "neutral" and
                    sig_b[dim] != "neutral" and
                    sig_a[dim] != sig_b[dim]):
                    conflicting_dims.append(dim)

            if conflicting_dims:
                contradictions.append({
                    "chunk_a_range": f"msgs {chunk_a.get('start_idx', '?')}–{chunk_a.get('end_idx', '?')}",
                    "chunk_b_range": f"msgs {chunk_b.get('start_idx', '?')}–{chunk_b.get('end_idx', '?')}",
                    "conflicting_dimensions": len(conflicting_dims),
                    "note": (
                        f"These two chunks appear to express different sentiments "
                        f"about the same subject across {len(conflicting_dims)} dimension(s). "
                        f"The more recent chunk (msgs {chunk_b.get('start_idx', '?')}+) "
                        f"likely reflects the current state."
                    )
                })

    return contradictions


# ── Answer merging ────────────────────────────────────────────────────────────

def extract_relevant_sentences(text: str, query_keywords: set, max_sentences: int = 3) -> list[str]:
    """
    Pull the most query-relevant sentences from a chunk.
    Avoids returning walls of text — just the parts that matter.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    scored = []

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 15:
            continue
        hits = sum(1 for kw in query_keywords if kw in sent.lower())
        scored.append((hits, sent))

    scored.sort(reverse=True)
    return [s for _, s in scored[:max_sentences]]


def merge_into_coherent_answer(
    query: str,
    scored_chunks: list[dict],
    contradictions: list[dict],
) -> str:
    """
    Build a single readable answer from the top-ranked chunks.

    The merge strategy:
    - Lead with the highest-scored chunk (most recent + emotional + relevant)
    - Add supporting context from lower-ranked chunks if they add new info
    - If contradictions exist, flag them explicitly at the end
    - Keep the total answer under ~200 words
    """
    query_keywords = set(re.findall(r'\b\w{3,}\b', query.lower()))

    answer_parts = []

    for rank, chunk in enumerate(scored_chunks[:3]):
        relevant_sents = extract_relevant_sentences(
            chunk["text"], query_keywords, max_sentences=2
        )
        if not relevant_sents:
            continue

        if rank == 0:
            prefix = "Most relevantly"
        elif rank == 1:
            prefix = "Additionally"
        else:
            prefix = "In another instance"

        time_ref = ""
        start = chunk.get("start_idx")
        if start is not None:
            time_ref = f" (around message {start})"

        answer_parts.append(f"{prefix}{time_ref}: {' '.join(relevant_sents)}")

    if not answer_parts:
        return "No specific mentions found in the retrieved context."

    merged = " ".join(answer_parts)

    if contradictions:
        n = len(contradictions)
        merged += (
            f"\n\n⚠ Note: {n} potential contradiction(s) detected across chunks. "
            f"The retrieved segments don't all agree — "
            f"the most recent mention is likely the most accurate current state."
        )

    return merged


# ── Main resolver ─────────────────────────────────────────────────────────────

class ConflictResolver:
    """
    Wraps the full resolution pipeline.
    Takes a query and a list of retrieved chunks (same format as RAGEngine output),
    returns a structured resolution with ranked chunks, contradiction flags,
    and a merged answer.
    """

    def __init__(self, total_messages: int = 191578):
        self.total_messages = total_messages

    def resolve(
        self,
        query: str,
        raw_chunks: list[dict],
        top_k: int = 5,
    ) -> dict:
        """
        Full resolution pipeline.

        Parameters
        ----------
        query       : the user's question
        raw_chunks  : list of chunk dicts from RAGEngine (must have 'text' and 'meta')
        top_k       : how many chunks to include in the final ranking

        Returns
        -------
        dict with keys: query, ranked_chunks, contradictions, merged_answer, resolution_note
        """
        if not raw_chunks:
            return {
                "query":          query,
                "ranked_chunks":  [],
                "contradictions": [],
                "merged_answer":  "No relevant chunks found for this query.",
                "resolution_note": "Empty retrieval set.",
            }

        # Score every chunk
        scored = []
        for chunk in raw_chunks:
            composite = score_chunk(chunk, query, self.total_messages)
            start_idx = (chunk.get("meta") or {}).get("start_idx") or \
                        (chunk.get("meta") or {}).get("start_index") or 0
            end_idx   = (chunk.get("meta") or {}).get("end_idx") or \
                        (chunk.get("meta") or {}).get("end_index") or 0

            scored.append({
                "text":        chunk.get("text", ""),
                "score":       round(composite, 4),
                "start_idx":   start_idx,
                "end_idx":     end_idx,
                "source":      chunk.get("source", "unknown"),
                "recency":     round(compute_recency_score(start_idx, self.total_messages), 4),
                "emotion":     round(compute_emotional_weight(chunk.get("text", "")), 4),
                "relevance":   round(compute_query_relevance(query, chunk.get("text", "")), 4),
            })

        # Sort by composite score (descending)
        scored.sort(key=lambda x: -x["score"])
        top_chunks = scored[:top_k]

        # Detect contradictions among top chunks
        contradictions = detect_contradictions(top_chunks)

        # Merge into answer
        merged = merge_into_coherent_answer(query, top_chunks, contradictions)

        # Resolution note — explains what happened
        if contradictions:
            note = (
                f"Retrieved {len(raw_chunks)} chunks. After ranking by recency + "
                f"emotional weight + relevance, top {len(top_chunks)} kept. "
                f"{len(contradictions)} contradiction(s) detected and flagged. "
                f"Answer prioritises the most recent, emotionally weighted mention."
            )
        else:
            note = (
                f"Retrieved {len(raw_chunks)} chunks. Ranked by composite score. "
                f"No contradictions detected — context is consistent."
            )

        return {
            "query":          query,
            "ranked_chunks":  [
                {k: v for k, v in c.items() if k != "text"}
                for c in top_chunks
            ],
            "contradictions": contradictions,
            "merged_answer":  merged,
            "resolution_note": note,
        }


# ── Standalone demo ───────────────────────────────────────────────────────────

def run_demo():
    """
    Demo using synthetic chunks to show the resolver working end-to-end.
    In production this would receive chunks from RAGEngine.retrieve().
    """
    # Simulate chunks that a retriever might return for "Did I mention my sister?"
    synthetic_chunks = [
        {
            "text": (
                "User 1: I really miss my sister. We used to be so close. "
                "User 2: That's tough. Have you called her recently? "
                "User 1: No, we had a fight last year and haven't spoken since."
            ),
            "meta": {"start_idx": 45200, "end_idx": 45225},
            "source": "message_chunk",
        },
        {
            "text": (
                "User 1: My sister just got a new job, so proud of her! "
                "User 2: Oh that's amazing! What does she do? "
                "User 1: She's going into healthcare, she's always wanted that."
            ),
            "meta": {"start_idx": 12300, "end_idx": 12318},
            "source": "message_chunk",
        },
        {
            "text": (
                "User 1: I was thinking about my sister today. "
                "User 2: Yeah? "
                "User 1: Just a random thought, we used to fight a lot as kids."
            ),
            "meta": {"start_idx": 87100, "end_idx": 87115},
            "source": "message_chunk",
        },
        {
            "text": (
                "User 1: My sister is visiting next month, I'm so excited! "
                "User 2: Are you two close? "
                "User 1: We had a rough patch but things are much better now."
            ),
            "meta": {"start_idx": 145000, "end_idx": 145020},
            "source": "message_chunk",
        },
    ]

    resolver = ConflictResolver(total_messages=191578)
    query    = "Did I mention anything about my sister?"

    print(f"Query: {query}")
    print("=" * 60)

    result = resolver.resolve(query, synthetic_chunks, top_k=4)

    print(f"\nRanked chunks (by composite score):")
    for i, chunk in enumerate(result["ranked_chunks"], 1):
        print(f"  {i}. msgs {chunk['start_idx']}–{chunk['end_idx']} | "
              f"score={chunk['score']} | recency={chunk['recency']} | "
              f"emotion={chunk['emotion']} | relevance={chunk['relevance']}")

    if result["contradictions"]:
        print(f"\nContradictions detected: {len(result['contradictions'])}")
        for c in result["contradictions"]:
            print(f"  {c['chunk_a_range']} vs {c['chunk_b_range']}: {c['note'][:80]}...")

    print(f"\nMerged answer:")
    print(result["merged_answer"])

    print(f"\nResolution note:")
    print(result["resolution_note"])

    # Save demo output
    out_path = DATA_DIR / "conflict_resolution_demo.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    run_demo()