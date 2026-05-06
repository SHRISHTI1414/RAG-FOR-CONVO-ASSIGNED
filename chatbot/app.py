"""
app.py — Streamlit chatbot
--------------------------
Ties together: RAG engine + persona + flan-t5-base generator.
Shows structured answers with source citations.

Run:  streamlit run chatbot/app.py
"""

import sys
import json
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

import streamlit as st
from rag_engine import RAGEngine, load_index
from generator import answer, answer_persona_question

DATA_DIR       = Path(__file__).parent.parent / "data"
PERSONA_PATH   = DATA_DIR / "persona.json"
CHECKPOINT_PATH = DATA_DIR / "topic_checkpoints.pkl"

st.set_page_config(
    page_title="Conversation RAG Chatbot",
    page_icon="💬",
    layout="wide",
)

# ── Resource loaders (cached) ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading RAG index (first run only)...")
def get_engine():
    return RAGEngine(load_index())

@st.cache_data
def get_persona() -> dict:
    with open(PERSONA_PATH) as f:
        return json.load(f)

@st.cache_data
def get_stats() -> dict:
    with open(CHECKPOINT_PATH, "rb") as f:
        data = pickle.load(f)
    return {
        "n_topics":  len(data["topic_checkpoints"]),
        "n_100msg":  len(data["message_checkpoints"]),
        "total_msgs": data["total_messages"],
        "sample_topics": data["topic_checkpoints"][:6],
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Index Stats")
    try:
        stats = get_stats()
        col1, col2 = st.columns(2)
        col1.metric("Total messages", f"{stats['total_msgs']:,}")
        col2.metric("Topics detected", stats["n_topics"])
        st.metric("100-msg checkpoints", stats["n_100msg"])

        st.divider()
        st.subheader("Sample Topic Segments")
        for cp in stats["sample_topics"]:
            with st.expander(
                f"Topic {cp['topic_id']} · msgs {cp['start_index']}–{cp['end_index']}"
            ):
                st.caption(f"**{cp['message_count']} messages** | "
                           f"boundary sim={cp.get('boundary_similarity', 'n/a')}")
                st.write("**Key phrases:**", ", ".join(cp.get("key_phrases", [])[:5]))
                st.write(cp["summary"][:250] + "...")
    except Exception as e:
        st.warning(f"Stats unavailable: {e}")

    st.divider()
    st.subheader("User Persona")
    try:
        persona = get_persona()
        high_habits = [
            k for k, v in persona.get("habits", {}).items()
            if v.get("confidence") == "high"
        ]
        high_traits = [
            k for k, v in persona.get("personality_traits", {}).items()
            if v.get("confidence") in ("high", "medium")
        ]
        style = persona.get("communication_style", {})

        st.write("**Top habits:**", ", ".join(high_habits[:5]) or "—")
        st.write("**Top traits:**", ", ".join(high_traits[:4]) or "—")
        st.write("**Tone:**", style.get("tone", "—"))
        st.write("**Msg style:**", style.get("style_label", "—"))
        st.write("**Occupations:**", ", ".join(persona.get("likely_occupations", [])[:3]))

        if st.toggle("Show full persona JSON"):
            st.json(persona)
    except Exception as e:
        st.warning(f"Persona unavailable: {e}")

    st.divider()
    st.caption("Stack: sentence-transformers · FAISS · flan-t5-base · No external API")


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("💬 Conversation RAG Chatbot")
st.caption(
    "Ask anything about the dataset. "
    "Every answer shows which topic segments and message chunks it was drawn from."
)

SUGGESTED = [
    "What kind of person is this user?",
    "What are their strongest habits?",
    "How do they communicate?",
    "What topics dominate the conversations?",
    "Does this user have pets?",
    "What jobs do people mention most?",
]

cols = st.columns(3)
for i, q in enumerate(SUGGESTED):
    if cols[i % 3].button(q, key=f"sug_{i}"):
        st.session_state["prefill"] = q

st.divider()

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant":
            st.markdown(turn["answer"])
            # Show structured sources
            sources = turn.get("sources", {})
            if sources.get("topics") or sources.get("chunks"):
                with st.expander("📚 Sources used"):
                    if sources.get("topics"):
                        st.markdown("**Topic segments:**")
                        for t in sources["topics"]:
                            st.caption(
                                f"Topic {t['topic_id']} · "
                                f"msgs {t['start_index']}–{t['end_index']} · "
                                f"similarity={t['score']}"
                            )
                    if sources.get("chunks"):
                        st.markdown("**Message chunks:**")
                        for c in sources["chunks"]:
                            st.caption(
                                f"msgs {c['start_idx']}–{c['end_idx']} · "
                                f"similarity={c['score']}"
                            )
            if turn.get("context"):
                with st.expander("🔍 Retrieved context"):
                    st.text(turn["context"])
        else:
            st.markdown(turn["content"])

# Input
prefill = st.session_state.pop("prefill", "")
query   = st.chat_input("Ask something about the conversations...") or prefill

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                persona = get_persona()
                engine  = get_engine()

                PERSONA_KEYWORDS = [
                    "kind of person", "habits", "how do they talk", "personality",
                    "communication style", "traits", "who is this user", "describe",
                    "emoji", "tone", "talk", "speak", "chat", "write",
                ]
                is_persona_q = any(kw in query.lower() for kw in PERSONA_KEYWORDS)

                if is_persona_q:
                    out      = answer_persona_question(query, persona)
                    context  = "(Answered directly from persona data)"
                else:
                    retrieval = engine.retrieve(query, top_k_topics=3, top_k_chunks=3)
                    context   = engine.build_context(retrieval)
                    out       = answer(query, context, retrieval, persona)

                st.markdown(out["answer"])

                # Structured sources panel
                sources = out.get("sources", {})
                if sources.get("topics") or sources.get("chunks"):
                    with st.expander("📚 Sources used"):
                        if sources.get("topics"):
                            st.markdown("**Topic segments:**")
                            for t in sources["topics"]:
                                st.caption(
                                    f"Topic {t['topic_id']} · "
                                    f"msgs {t['start_index']}–{t['end_index']} · "
                                    f"similarity={t['score']}"
                                )
                        if sources.get("chunks"):
                            st.markdown("**Message chunks:**")
                            for c in sources["chunks"]:
                                st.caption(
                                    f"msgs {c['start_idx']}–{c['end_idx']} · "
                                    f"similarity={c['score']}"
                                )

                with st.expander("🔍 Retrieved context"):
                    st.text(context)

                st.caption(f"_Model: {out.get('model_used', '?')}_")

                st.session_state.history.append({
                    "role":    "assistant",
                    "answer":  out["answer"],
                    "sources": out.get("sources", {}),
                    "context": context,
                })

            except Exception as e:
                msg = f"Error: {e}"
                st.error(msg)
                st.session_state.history.append({
                    "role": "assistant", "answer": msg, "sources": {}
                })

if st.session_state.history and st.button("Clear chat"):
    st.session_state.history = []
    st.rerun()
