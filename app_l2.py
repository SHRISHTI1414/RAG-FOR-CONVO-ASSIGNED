import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "core"))
sys.path.insert(0, str(Path(__file__).parent / "core" / "drift"))
sys.path.insert(0, str(Path(__file__).parent / "core" / "intent"))
sys.path.insert(0, str(Path(__file__).parent / "core" / "conflict"))
import streamlit as st
DATA_DIR = Path(__file__).parent / "data"
st.set_page_config(page_title="Conversation Intelligence L2", page_icon="🧠", layout="wide")
st.title("🧠 Conversation Intelligence System — L2")
st.caption("Adaptive Persona Engine · Intent Classifier · Conflict Resolver · System Design")
tab1, tab2, tab3, tab4 = st.tabs(["📈 Persona Drift", "🎯 Intent Classifier", "⚖️ Conflict Resolver", "🏗️ System Design"])

with tab1:
    st.header("Persona Drift — Mood Timeline Across Days")
    st.caption("Tracks how the user's emotional tone shifts day by day, and what triggered each shift.")
    try:
        report = json.load(open(DATA_DIR / "drift_report.json"))
        col1, col2, col3 = st.columns(3)
        col1.metric("Days Analysed", report["total_days_analysed"])
        col2.metric("Drift Events", report["total_drift_events"])
        col3.metric("Baseline Tone", report["persona_baseline"].get("tone", "—"))
        st.divider()
        st.subheader("Overall Arc")
        st.info(report["arc_summary"])
        st.subheader("Drift Events")
        mood_colours = {"curious":"🔵","casual":"🟢","playful":"🟡","emotional":"🟠","frustrated":"🔴","formal":"⚪"}
        for ev in report["drift_events"][:15]:
            with st.expander(f"Day {ev['from_day']} → {ev['to_day']} | {ev['from_mood']} → {ev['to_mood']} | {ev['likely_trigger'].replace('_',' ')}"):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Magnitude", ev["drift_magnitude"])
                col_b.metric("Energy Before", ev["energy_before"])
                col_c.metric("Energy After", ev["energy_after"])
                st.write("**Interpretation:**", ev["interpretation"])
                if ev.get("trigger_keywords"):
                    st.write("**Keywords:**", ", ".join(ev["trigger_keywords"]))
        st.divider()
        st.subheader("Day-by-Day Timeline (first 30)")
        for d in report.get("mood_timeline", [])[:30]:
            icon = mood_colours.get(d["mood"], "⬜")
            drift = " 🔀 drift point" if d.get("is_drift_point") else ""
            st.write(f"**Day {d['day']}** {icon} `{d['mood']}` — topic: {d['dominant_topic'].replace('_',' ')} | energy: {d['energy']}{drift}")
    except FileNotFoundError:
        st.warning("Run core/drift/drift_detector.py first.")
with tab2:
    st.header("Intent Classifier — Offline, <1ms, <150KB")
    st.caption("Classify any message into: reminder / emotional_support / action_item / small_talk / unknown")
    try:
        from intent_classifier import IntentClassifier
        clf = IntentClassifier(DATA_DIR / "intent_classifier.pkl")
        user_msg = st.text_input("Type a message:", placeholder="e.g. don't forget to call mom tomorrow")
        if user_msg:
            result = clf.classify(user_msg)
            col1, col2, col3 = st.columns(3)
            col1.metric("Intent", result["intent"].replace("_"," ").title())
            col2.metric("Confidence", f"{result['confidence']*100:.1f}%")
            col3.metric("Latency", f"{result['latency_ms']:.2f}ms")
            st.subheader("All class probabilities")
            for intent, score in result["all_scores"].items():
                bar = "█" * int(score*30) + "░" * (30 - int(score*30))
                st.text(f"{intent:>20} {bar} {score:.3f}")
        st.divider()
        st.subheader("Corpus Intent Distribution")
        try:
            analysis = json.load(open(DATA_DIR / "intent_analysis.json"))
            for intent, count in analysis["intent_distribution"].items():
                st.write(f"**{intent.replace('_',' ').title()}**: {count} msgs ({analysis['intent_percentages'][intent]}%)")
            st.caption(f"Avg latency: {analysis['avg_latency_ms']}ms across {analysis['messages_classified']:,} messages")
        except FileNotFoundError:
            st.info("Run core/intent/intent_classifier.py for corpus analysis.")
    except Exception as e:
        st.error(f"Error: {e}")
with tab3:
    st.header("Conflict Resolver — Handles Contradictory RAG Chunks")
    st.caption("Ranks by recency + emotional weight + relevance, detects contradictions, merges into coherent answer.")
    try:
        from conflict_resolver import ConflictResolver
        resolver = ConflictResolver(total_messages=191578)
        query = st.text_input("Query:", value="Did I mention anything about my sister?")
        demo_chunks = [
            {"text": "User 1: I really miss my sister. We used to be so close. User 1: we had a fight last year and haven't spoken since.", "meta": {"start_idx": 45200, "end_idx": 45225}, "source": "message_chunk"},
            {"text": "User 1: My sister just got a new job, so proud of her! User 1: She's going into healthcare.", "meta": {"start_idx": 12300, "end_idx": 12318}, "source": "message_chunk"},
            {"text": "User 1: I was thinking about my sister today. User 1: we used to fight a lot as kids.", "meta": {"start_idx": 87100, "end_idx": 87115}, "source": "message_chunk"},
            {"text": "User 1: My sister is visiting next month, I'm so excited! User 1: We had a rough patch but things are much better now.", "meta": {"start_idx": 145000, "end_idx": 145020}, "source": "message_chunk"},
        ]
        if st.button("Resolve", type="primary"):
            result = resolver.resolve(query, demo_chunks, top_k=4)
            st.subheader("Ranked Chunks")
            for i, c in enumerate(result["ranked_chunks"], 1):
                st.write(f"**#{i}** msgs {c['start_idx']}–{c['end_idx']} | score={c['score']} | recency={c['recency']} | emotion={c['emotion']} | relevance={c['relevance']}")
            if result["contradictions"]:
                st.subheader(f"⚠️ {len(result['contradictions'])} Contradiction(s) Detected")
                for c in result["contradictions"]:
                    st.warning(f"{c['chunk_a_range']} vs {c['chunk_b_range']}: {c['note']}")
            else:
                st.success("No contradictions — chunks are consistent.")
            st.subheader("Merged Answer")
            st.write(result["merged_answer"])
            st.caption(result["resolution_note"])
    except Exception as e:
        st.error(f"Error: {e}")

with tab4:
    st.header("System Design — Sync Architecture")
    try:
        st.markdown(open(Path(__file__).parent / "SYSTEM_DESIGN.md").read())
    except FileNotFoundError:
        st.error("SYSTEM_DESIGN.md not found.")