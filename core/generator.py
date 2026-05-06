"""
generator.py
------------
Takes a query + retrieved context and generates a structured answer.

Output format:
{
    "answer":  str,           # natural language answer
    "sources": {
        "topics": [           # which topic segments contributed
            {"topic_id": 3, "start_index": 240, "end_index": 510, "score": 0.81}
        ],
        "chunks": [           # which message chunks contributed
            {"start_idx": 245, "end_idx": 269, "score": 0.76}
        ]
    },
    "model_used": str         # "flan-t5-base" or "rule-based"
}

Model: google/flan-t5-base (250M params, CPU-friendly, ~2-4s per answer).
Falls back to rule-based keyword extraction if model can't be downloaded.
"""

import re
from pathlib import Path

_generator   = None
_use_fallback = False


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model():
    global _generator, _use_fallback
    if _generator is not None or _use_fallback:
        return _generator
    try:
        from transformers import pipeline
        print("[generator] Loading flan-t5-base...")
        _generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256,
            do_sample=False,
        )
        print("[generator] Model ready")
    except Exception as e:
        print(f"[generator] flan-t5 unavailable ({e}) — using rule-based fallback")
        _use_fallback = True
    return _generator


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _rule_based(query: str, context: str, persona: dict = None) -> str:
    """
    When no LLM is available: score context sentences by keyword overlap
    with the query, return the top sentences as the answer.
    """
    keywords = [w for w in re.split(r"\W+", query.lower()) if len(w) > 3]
    sentences = re.split(r"(?<=[.!?])\s+", context)

    scored = []
    for s in sentences:
        s = s.strip()
        if len(s) < 10:
            continue
        score = sum(1 for k in keywords if k in s.lower())
        if score > 0:
            scored.append((score, s))

    scored.sort(reverse=True)
    top = [s for _, s in scored[:4]]

    if top:
        return " ".join(top)

    # Last resort: pull from persona
    if persona:
        habits = ", ".join(
            k for k, v in persona.get("habits", {}).items()
            if v.get("confidence") in ("high", "medium")
        )
        traits = ", ".join(
            k for k, v in persona.get("personality_traits", {}).items()
            if v.get("confidence") in ("high", "medium")
        )
        return f"Based on the conversations: habits include {habits}; personality traits include {traits}."

    return "I couldn't find a specific answer in the retrieved context."


# ── Core answer functions ─────────────────────────────────────────────────────

def answer(query: str, context: str, retrieval: dict, persona: dict = None) -> dict:
    """
    Generate a structured answer from context + retrieval metadata.

    Parameters
    ----------
    query     : user's question
    context   : formatted context string (from RAGEngine.build_context)
    retrieval : raw retrieval dict (from RAGEngine.retrieve) — used for sources
    persona   : parsed persona JSON (optional, enriches the answer)

    Returns
    -------
    dict with keys: answer, sources, model_used
    """
    gen = _load_model()

    # Build sources citation
    sources = {
        "topics": [
            {
                "topic_id":    h["meta"].get("topic_id"),
                "start_index": h["meta"].get("start_index"),
                "end_index":   h["meta"].get("end_index"),
                "score":       h["score"],
            }
            for h in retrieval.get("topic_summaries", [])
        ],
        "chunks": [
            {
                "start_idx": h["meta"].get("start_idx"),
                "end_idx":   h["meta"].get("end_idx"),
                "score":     h["score"],
            }
            for h in retrieval.get("message_chunks", [])
        ],
    }

    # Persona hint for context
    persona_hint = ""
    if persona:
        style  = persona.get("communication_style", {})
        habits = ", ".join(
            k for k, v in persona.get("habits", {}).items()
            if v.get("confidence") == "high"
        )
        traits = ", ".join(
            k for k, v in persona.get("personality_traits", {}).items()
            if v.get("confidence") in ("high", "medium")
        )
        persona_hint = (
            f"\nUser persona: tone={style.get('tone', 'unknown')}, "
            f"habits={habits}, traits={traits}."
        )

    if _use_fallback or gen is None:
        text = _rule_based(query, context, persona)
        return {"answer": text, "sources": sources, "model_used": "rule-based"}

    prompt = (
        f"Answer the question based only on the context provided.\n\n"
        f"Context:\n{context[:1400]}\n"
        f"{persona_hint}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    result = gen(prompt)
    text   = result[0]["generated_text"].strip()
    return {"answer": text, "sources": sources, "model_used": "flan-t5-base"}


def answer_persona_question(query: str, persona: dict) -> dict:
    """
    Answer questions that are specifically about the user's persona.
    Uses structured persona data directly — no RAG needed.
    """
    gen = _load_model()

    style  = persona.get("communication_style", {})
    habits = {
        k: f"{v['confidence']} confidence ({v['hit_count']} signals)"
        for k, v in persona.get("habits", {}).items()
    }
    traits = {
        k: f"{v['confidence']} confidence, e.g. \"{v['evidence'][0][:60]}...\""
        for k, v in persona.get("personality_traits", {}).items()
        if v.get("evidence")
    }
    rels   = list(persona.get("relationships", {}).keys())
    jobs   = persona.get("likely_occupations", [])

    persona_str = (
        f"Habits: {habits}. "
        f"Personality traits: {traits}. "
        f"Relationships: {rels}. "
        f"Occupations mentioned: {jobs}. "
        f"Communication style: {style.get('style_label')}, tone={style.get('tone')}, "
        f"emoji usage={style.get('emoji_usage')}."
    )

    if _use_fallback or gen is None:
        # Build a direct answer from the structured data
        top_habits = [k for k, v in persona.get("habits", {}).items()
                      if v.get("confidence") == "high"][:4]
        top_traits = [k for k, v in persona.get("personality_traits", {}).items()
                      if v.get("confidence") in ("high", "medium")][:4]

        if "habit" in query.lower():
            text = f"This user's strongest habits are: {', '.join(top_habits)}."
        elif "talk" in query.lower() or "communicat" in query.lower() or "style" in query.lower():
            text = (
                f"They write {style.get('style_label')} messages with a {style.get('tone')} tone. "
                f"Emoji usage is {style.get('emoji_usage')}. "
                f"They ask questions {round(style.get('question_ratio', 0)*100)}% of the time."
            )
        elif "person" in query.lower() or "kind" in query.lower() or "who" in query.lower():
            sample_traits = [
                f"{k} (e.g. \"{v['evidence'][0][:50]}\")"
                for k, v in list(persona.get("personality_traits", {}).items())[:3]
                if v.get("evidence")
            ]
            text = f"This user comes across as: {'; '.join(sample_traits)}."
        else:
            text = f"Habits: {', '.join(top_habits)}. Traits: {', '.join(top_traits)}."

        return {"answer": text, "sources": {"topics": [], "chunks": []}, "model_used": "rule-based"}

    prompt = (
        f"Based on this user profile, answer the question.\n\n"
        f"Profile: {persona_str[:800]}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    result = gen(prompt)
    text   = result[0]["generated_text"].strip()
    return {"answer": text, "sources": {"topics": [], "chunks": []}, "model_used": "flan-t5-base"}


if __name__ == "__main__":
    _load_model()
    ctx = "User 1 loves reading books. User 1 goes to the gym every morning. Has a dog named Buddy."
    fake_retrieval = {"topic_summaries": [], "message_chunks": []}
    out = answer("What are the user's habits?", ctx, fake_retrieval)
    print(out)
