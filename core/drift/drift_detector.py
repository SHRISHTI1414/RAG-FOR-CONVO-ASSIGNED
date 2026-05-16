import re, json, math
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent / "data"

TONE_LEXICONS = {
    "formal": ["however","therefore","furthermore","indeed","regarding","appreciate","certainly","absolutely","sincere","professional"],
    "casual": ["yeah","yep","wanna","gonna","kinda","btw","lol","haha","omg","cool","awesome","totally","literally","guys"],
    "frustrated": ["annoying","frustrating","ugh","terrible","awful","hate","worst","ridiculous","exhausted","tired","sick","disappointed"],
    "curious": ["wonder","curious","interesting","fascinating","tell me more","really","what if","have you ever","question","explain"],
    "playful": ["haha","lol","funny","joke","kidding","silly","fun","exciting","amazing","wow","hilarious","laughing"],
    "emotional": ["feel","miss","sad","happy","love","hurt","scared","worried","nervous","proud","grateful","lonely","overwhelmed"],
    "positive": ["great","good","wonderful","excellent","enjoy","happy","glad","pleased","thrilled","fantastic","perfect","blessed"],
    "negative": ["bad","terrible","horrible","awful","hate","sad","unhappy","upset","angry","frustrated","disappointed","struggle"]
}

TRIGGER_TOPICS = {
    "work_stress":   ["work","job","boss","deadline","office","meeting","project","career"],
    "relationships": ["friend","boyfriend","girlfriend","family","sister","brother","mom","dad","partner"],
    "health":        ["sick","tired","sleep","doctor","hospital","health","pain","exercise"],
    "money":         ["money","rent","bills","expensive","afford","salary","broke","budget"],
    "life_change":   ["moving","new job","breakup","graduation","wedding","baby","divorce","change"],
    "hobbies":       ["reading","cooking","music","travel","gym","yoga","game","art","sport"],
    "social":        ["party","friends","hangout","date","event","celebrate","lonely","alone"],
}

def extract_user1_text(raw):
    lines = raw.split("\n")
    out = []
    for line in lines:
        line = line.strip()
        if line.startswith("User 1:"):
            t = line[len("User 1:"):].strip()
            if t:
                out.append(t)
    return " ".join(out)

def score_tone_axes(text):
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return {ax: 0.0 for ax in TONE_LEXICONS}
    wc = len(words)
    full = text.lower()
    scores = {}
    for ax, lexicon in TONE_LEXICONS.items():
        hits = sum(1 for term in lexicon if term in full)
        scores[ax] = min(hits / max(wc / 100, 1), 1.0)
    return scores

def dominant_mood(scores):
    mood_map = {k: scores.get(k, 0) for k in ["curious","formal","casual","frustrated","playful","emotional"]}
    neg_gap = scores.get("negative", 0) - scores.get("positive", 0)
    if neg_gap > 0.15:
        mood_map["frustrated"] += neg_gap
    return max(mood_map, key=mood_map.get)

def detect_trigger(text):
    full = text.lower()
    counts = {}
    words_matched = {}
    for topic, kws in TRIGGER_TOPICS.items():
        matched = [kw for kw in kws if kw in full]
        if matched:
            counts[topic] = len(matched)
            words_matched[topic] = matched
    if not counts:
        return "general_conversation", []
    best = max(counts, key=counts.get)
    return best, words_matched[best][:4]

def energy_score(text):
    if not text:
        return 0.0
    sents = re.split(r'[.!?]', text)
    n = max(len(sents), 1)
    return round((text.count('!') / n * 0.4 + text.count('?') / n * 0.3), 4)

def drift_magnitude(a, b):
    axes = list(TONE_LEXICONS.keys())
    va = np.array([a.get(ax, 0) for ax in axes])
    vb = np.array([b.get(ax, 0) for ax in axes])
    return float(np.linalg.norm(vb - va))

def interpret_drift(fm, tm, trigger):
    shift_map = {
        ("curious","frustrated"): "User shifted from exploration to friction — possibly unmet expectations.",
        ("formal","casual"): "Conversation relaxed noticeably — user felt more comfortable.",
        ("casual","frustrated"): "Light mood turned heavier — something struck a nerve.",
        ("playful","emotional"): "Shifted from light-hearted to introspective — personal topics surfacing.",
        ("emotional","playful"): "Recovery pattern — user bounced back after a heavier period.",
        ("frustrated","casual"): "Tension eased — topic resolved or user disengaged from stressor.",
        ("curious","playful"): "Curiosity converted into enthusiasm.",
        ("formal","emotional"): "Formal tone broke into something personal — guard came down.",
    }
    desc = shift_map.get((fm, tm), f"Shift from {fm} to {tm}.")
    return f"{desc} Likely connected to: {trigger.replace('_',' ')}."

def build_profiles(df, sample_every=10):
    profiles = []
    for i in range(0, len(df), sample_every):
        raw  = df.iloc[i]["conversation"]
        text = extract_user1_text(raw)
        if len(text.strip()) < 20:
            continue
        scores  = score_tone_axes(text)
        mood    = dominant_mood(scores)
        trig, kws = detect_trigger(text)
        profiles.append({
            "day": i + 1,
            "mood": mood,
            "tone_scores": scores,
            "trigger_topic": trig,
            "trigger_keywords": kws,
            "energy_level": energy_score(text),
            "text_length": len(text.split()),
        })
    return profiles

def find_drifts(profiles, threshold=0.18, min_gap=3):
    events = []
    last_drift = -min_gap
    for i in range(1, len(profiles)):
        prev = profiles[i-1]
        curr = profiles[i]
        mag  = drift_magnitude(prev["tone_scores"], curr["tone_scores"])
        gap  = curr["day"] - last_drift
        if mag >= threshold and gap >= min_gap:
            events.append({
                "from_day": prev["day"],
                "to_day": curr["day"],
                "from_mood": prev["mood"],
                "to_mood": curr["mood"],
                "drift_magnitude": round(mag, 4),
                "likely_trigger": curr["trigger_topic"],
                "trigger_keywords": curr["trigger_keywords"],
                "energy_before": prev["energy_level"],
                "energy_after": curr["energy_level"],
                "interpretation": interpret_drift(prev["mood"], curr["mood"], curr["trigger_topic"]),
            })
            last_drift = curr["day"]
    return events

def arc_summary(events):
    if not events:
        return "No significant mood shifts detected."
    moods = [events[0]["from_mood"]] + [e["to_mood"] for e in events]
    arc   = " → ".join(dict.fromkeys(moods))
    top   = Counter(e["likely_trigger"] for e in events).most_common(1)[0][0].replace("_"," ")
    return (f"Across {len(events)} drift point(s), emotional arc: {arc}. "
            f"Most common trigger: '{top}'. "
            f"Pattern suggests {'high emotional variability' if len(events) > 5 else 'relative stability with occasional shifts'}.")

def run(csv_path=None, persona_path=None, sample_every=10, output_path=None):
    csv_path     = csv_path     or DATA_DIR / "conversations.csv"
    persona_path = persona_path or DATA_DIR / "persona.json"
    output_path  = output_path  or DATA_DIR / "drift_report.json"

    df      = pd.read_csv(csv_path, header=0)
    df.columns = ["conversation"]
    persona = json.load(open(persona_path))

    print(f"Profiling days (sample_every={sample_every})...")
    profiles = build_profiles(df, sample_every)
    print(f"  {len(profiles)} day profiles built")

    print("Detecting drifts...")
    events = find_drifts(profiles)
    print(f"  {len(events)} drift events found")

    timeline = []
    drift_days = {e["to_day"] for e in events}
    for p in profiles:
        entry = {"day": p["day"], "mood": p["mood"], "energy": p["energy_level"],
                 "dominant_topic": p["trigger_topic"], "is_drift_point": p["day"] in drift_days}
        if p["day"] in drift_days:
            match = [e for e in events if e["to_day"] == p["day"]]
            if match:
                entry["drift_info"] = match[0]
        timeline.append(entry)

    report = {
        "total_days_analysed": len(profiles),
        "total_drift_events":  len(events),
        "arc_summary":         arc_summary(events),
        "drift_events":        events,
        "mood_timeline":       timeline[:50],
        "persona_baseline": {
            "established_traits": list(persona.get("personality_traits", {}).keys()),
            "tone": persona.get("communication_style", {}).get("tone", "unknown"),
        }
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved → {output_path}")
    return report

if __name__ == "__main__":
    r = run(sample_every=10)
    print("\nArc:", r["arc_summary"])
    print("\nFirst 3 drifts:")
    for e in r["drift_events"][:3]:
        print(f"  Day {e['from_day']}→{e['to_day']}: {e['from_mood']}→{e['to_mood']} | {e['likely_trigger']}")
        print(f"  {e['interpretation']}")
