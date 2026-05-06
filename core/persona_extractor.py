"""
persona_extractor.py
--------------------
Extracts a structured, evidence-backed persona for "User 1" from
across all conversations.

Design principle: every trait must be grounded in actual message text.
No guessing, no hallucination — each signal stores the exact lines that
triggered it.

Output format:
{
  "habits": {
    "reader": {
      "confidence": "high",         # high / medium / low
      "hit_count": 487,
      "evidence": ["I love reading", "I've been reading a lot lately"]
    },
    ...
  },
  "personality_traits": { ... same structure ... },
  "relationships": { ... },
  "communication_style": {
    "avg_message_length_words": 11.1,
    "style_label": "moderate",
    "tone": "enthusiastic",
    "emoji_usage": "rare",
    "question_ratio": 0.27,
    "exclamation_ratio": 0.44
  },
  "likely_occupations": ["teacher", "writer", ...],
  "locations_mentioned": ["california", "the midwest", ...],
  "total_messages_analysed": 98072
}
"""

import re
import json
from collections import Counter, defaultdict
from pathlib import Path

from parser import load_messages

PERSONA_PATH = Path(__file__).parent.parent / "data" / "persona.json"

# ── Pattern banks ─────────────────────────────────────────────────────────────
# Each entry: pattern_name → list of regex strings to match against User 1 text.
# We store up to MAX_EVIDENCE actual lines as evidence per trait.

MAX_EVIDENCE = 3  # how many example lines to keep per trait

HABIT_PATTERNS: dict[str, list[str]] = {
    "reader": [
        r"\b(love|like|enjoy|adore)\s+(to\s+)?read(ing)?\b",
        r"\b(reading|book|novel|fiction|nonfiction)\b",
        r"\bjust\s+finished\s+(reading|a\s+book)\b",
    ],
    "cook": [
        r"\b(love|like|enjoy)\s+(to\s+)?(cook|bake|make\s+food)\b",
        r"\b(cooking|baking|recipe|made\s+dinner|meal)\b",
    ],
    "fitness": [
        r"\b(gym|workout|work\s+out|exercise|run(ning)?|jog(ging)?|yoga|lift(ing)?)\b",
        r"\bstay(ing)?\s+(in\s+shape|fit|active)\b",
    ],
    "music_lover": [
        r"\b(love|like|enjoy)\s+(music|listening\s+to)\b",
        r"\b(play(ing)?|guitar|piano|drums|violin|bass|sing(ing)?|band)\b",
    ],
    "gamer": [
        r"\b(video\s+games?|gaming|play\s+games?|gamer|xbox|playstation|nintendo|pc\s+gaming)\b",
    ],
    "traveler": [
        r"\b(travel(ing)?|hike|hiking|backpack(ing)?|road\s+trip|vacation|explore)\b",
        r"\b(been\s+to|visited|trip\s+to|flew\s+to)\b",
    ],
    "pet_owner": [
        r"\bmy\s+(dog|cat|pet|puppy|kitten|rabbit|hamster)\b",
        r"\bi\s+have\s+a\s+(dog|cat|pet)\b",
    ],
    "coffee_drinker": [
        r"\b(coffee|latte|espresso|cappuccino|cold\s+brew|caffeine)\b",
    ],
    "early_riser": [
        r"\b(wake\s+up\s+early|morning\s+person|up\s+at\s+\d\s*am|early\s+morning)\b",
    ],
    "late_sleeper": [
        r"\b(stay\s+up\s+late|night\s+owl|up\s+all\s+night|can'?t\s+sleep|insomnia|awake\s+at\s+\d+\s*am)\b",
    ],
    "social": [
        r"\b(hang(ing)?\s+out|meet\s+(up|new\s+people)|friends?|party|sociali[sz]e)\b",
    ],
}

TRAIT_PATTERNS: dict[str, list[str]] = {
    "empathetic": [
        r"\b(sorry\s+to\s+hear|that\s+must\s+be\s+(hard|tough|difficult)|i\s+understand|feel\s+for\s+you)\b",
        r"\b(hope\s+you('?re|\s+are)\s+(ok|okay|alright|feeling\s+better))\b",
    ],
    "humorous": [
        r"\b(haha|hehe|lol|lmao|lmfao|funny|hilarious|joke|just\s+kidding|jk)\b",
        r"😂|🤣|😄",
    ],
    "curious": [
        r"\b(tell\s+me\s+more|how\s+does\s+that\s+work|why\s+do\s+you|i'?m\s+curious|interested\s+in|always\s+wondered)\b",
        r"\bwhat\s+(do|did|does|is|are|was|were)\b.{0,40}\?",
    ],
    "optimistic": [
        r"\b(can'?t\s+wait|so\s+excited|looking\s+forward|can'?t\s+wait|love\s+that|that'?s\s+(amazing|awesome|great|fantastic|wonderful))\b",
    ],
    "emotional": [
        r"\b(i\s+feel|i\s+miss|so\s+sad|makes\s+me\s+(happy|sad|cry|emotional)|i\s+cried|broke\s+my\s+heart)\b",
    ],
    "ambitious": [
        r"\b(goal(s)?|dream|want\s+to\s+become|working\s+(toward|towards)|career|aspire|achieve|one\s+day\s+i'?ll)\b",
    ],
    "introverted": [
        r"\b(stay\s+(at\s+)?home|alone\s+time|need\s+my\s+space|recharge|quiet\s+night|by\s+myself)\b",
    ],
    "extroverted": [
        r"\b(love\s+meeting\s+new\s+people|love\s+parties|being\s+around\s+people|social\s+(butterfly|person))\b",
    ],
    "thoughtful": [
        r"\b(i\s+think|i\s+believe|in\s+my\s+opinion|i\s+feel\s+like|from\s+my\s+perspective|i\s+wonder)\b",
    ],
}

RELATIONSHIP_PATTERNS: dict[str, list[str]] = {
    "has_kids": [
        r"\bmy\s+(kid|child|son|daughter|children|little\s+one)\b",
        r"\bi('?m|\s+am)\s+a\s+(mom|dad|mother|father|parent)\b",
    ],
    "has_siblings": [
        r"\bmy\s+(brother|sister|sibling|bro|sis)\b",
    ],
    "has_romantic_partner": [
        r"\bmy\s+(boyfriend|girlfriend|husband|wife|partner|spouse|fianc[eé])\b",
    ],
    "is_single": [
        r"\bi('?m|\s+am)\s+single\b",
        r"\b(not\s+dating|no\s+boyfriend|no\s+girlfriend)\b",
    ],
    "has_parents_in_life": [
        r"\bmy\s+(mom|dad|mother|father|parents?)\b",
    ],
}

JOB_PATTERNS = [
    r"i(?:'m|\s+am)\s+a\s+([\w][\w\s]{1,25}?)(?:\.|,|!|\band\b|$)",
    r"i\s+work\s+as\s+a\s+([\w][\w\s]{1,25}?)(?:\.|,|!|$)",
    r"i\s+work\s+(in|at)\s+([\w][\w\s]{1,25}?)(?:\.|,|!|$)",
    r"my\s+job\s+is\s+([\w][\w\s]{1,25}?)(?:\.|,|!|$)",
    r"i(?:'m|\s+am)\s+(?:studying|a\s+student\s+of)\s+([\w][\w\s]{1,25}?)(?:\.|,|!|$)",
]

LOCATION_PATTERNS = [
    r"i(?:'m|\s+am)\s+(?:from|in|living\s+in|based\s+in)\s+([\w][\w\s,]{2,30}?)(?:\.|,|!|$)",
    r"i\s+live\s+in\s+([\w][\w\s]{2,25}?)(?:\.|,|!|$)",
    r"i(?:'m|\s+am)\s+moving\s+to\s+([\w][\w\s,]{2,30}?)(?:\.|,|!|$)",
]

EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    r"\u2600-\u26FF\u2700-\u27BF]+",
    flags=re.UNICODE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _match(text: str, patterns: list[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t, re.IGNORECASE) for p in patterns)


def _confidence(count: int) -> str:
    if count >= 20:
        return "high"
    if count >= 5:
        return "medium"
    return "low"


def _regex_values(text: str, patterns: list[str]) -> list[str]:
    results = []
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            # some patterns have 2 groups (e.g. "work in X")
            val = m.group(m.lastindex or 1).strip().rstrip(".,!")
            if 2 < len(val) < 45:
                results.append(val.lower())
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def extract_persona(messages: list[dict] = None) -> dict:
    if messages is None:
        messages = load_messages()

    user1 = [m for m in messages if m["speaker"] == "User 1"]
    print(f"[persona] Analysing {len(user1):,} User 1 messages")

    # ── Habits ────────────────────────────────────────────────────────────────
    habits: dict[str, dict] = {}
    habit_hits   = defaultdict(int)
    habit_evidence = defaultdict(list)

    for m in user1:
        for name, patterns in HABIT_PATTERNS.items():
            if _match(m["text"], patterns):
                habit_hits[name] += 1
                if len(habit_evidence[name]) < MAX_EVIDENCE:
                    habit_evidence[name].append(m["text"])

    for name, count in sorted(habit_hits.items(), key=lambda x: -x[1]):
        if count >= 2:
            habits[name] = {
                "confidence": _confidence(count),
                "hit_count": count,
                "evidence": habit_evidence[name],
            }

    # ── Traits ────────────────────────────────────────────────────────────────
    traits: dict[str, dict] = {}
    trait_hits     = defaultdict(int)
    trait_evidence = defaultdict(list)

    for m in user1:
        for name, patterns in TRAIT_PATTERNS.items():
            if _match(m["text"], patterns):
                trait_hits[name] += 1
                if len(trait_evidence[name]) < MAX_EVIDENCE:
                    trait_evidence[name].append(m["text"])

    for name, count in sorted(trait_hits.items(), key=lambda x: -x[1]):
        if count >= 2:
            traits[name] = {
                "confidence": _confidence(count),
                "hit_count": count,
                "evidence": trait_evidence[name],
            }

    # ── Relationships ─────────────────────────────────────────────────────────
    relationships: dict[str, dict] = {}
    rel_hits     = defaultdict(int)
    rel_evidence = defaultdict(list)

    for m in user1:
        for name, patterns in RELATIONSHIP_PATTERNS.items():
            if _match(m["text"], patterns):
                rel_hits[name] += 1
                if len(rel_evidence[name]) < MAX_EVIDENCE:
                    rel_evidence[name].append(m["text"])

    for name, count in rel_hits.items():
        if count >= 1:
            relationships[name] = {
                "confidence": _confidence(count),
                "hit_count": count,
                "evidence": rel_evidence[name],
            }

    # ── Occupations ───────────────────────────────────────────────────────────
    job_mentions: list[str] = []
    for m in user1:
        job_mentions.extend(_regex_values(m["text"], JOB_PATTERNS))

    job_counter = Counter(job_mentions)
    occupations = [j for j, _ in job_counter.most_common(5) if len(j.split()) <= 4]

    # ── Locations ─────────────────────────────────────────────────────────────
    loc_mentions: list[str] = []
    for m in user1:
        loc_mentions.extend(_regex_values(m["text"], LOCATION_PATTERNS))

    loc_counter = Counter(loc_mentions)
    locations   = [l for l, _ in loc_counter.most_common(5)]

    # ── Communication style ───────────────────────────────────────────────────
    lengths    = [len(m["text"].split()) for m in user1]
    avg_len    = round(sum(lengths) / len(lengths), 1)

    emoji_c    = sum(1 for m in user1 if EMOJI_RE.search(m["text"]))
    question_c = sum(1 for m in user1 if "?" in m["text"])
    exclaim_c  = sum(1 for m in user1 if "!" in m["text"])

    n = len(user1)
    q_ratio  = round(question_c / n, 3)
    ex_ratio = round(exclaim_c  / n, 3)
    em_ratio = round(emoji_c    / n, 3)

    style_label = (
        "short and punchy" if avg_len < 8
        else "long and detailed" if avg_len > 18
        else "moderate length"
    )

    tone = (
        "enthusiastic and expressive" if ex_ratio > 0.4
        else "inquisitive and conversational" if q_ratio > 0.4
        else "casual (emoji-heavy)" if em_ratio > 0.1
        else "calm and straightforward"
    )

    style = {
        "avg_message_length_words": avg_len,
        "style_label": style_label,
        "tone": tone,
        "emoji_usage":  "frequent" if em_ratio > 0.1 else "occasional" if em_ratio > 0.02 else "rare",
        "emoji_ratio":  em_ratio,
        "question_ratio":     q_ratio,
        "exclamation_ratio":  ex_ratio,
    }

    # ── Final persona ─────────────────────────────────────────────────────────
    persona = {
        "habits":                habits,
        "personality_traits":    traits,
        "relationships":         relationships,
        "communication_style":   style,
        "likely_occupations":    occupations,
        "locations_mentioned":   locations,
        "total_messages_analysed": len(user1),
    }
    return persona


def run_and_save() -> dict:
    messages = load_messages()
    persona  = extract_persona(messages)
    with open(PERSONA_PATH, "w") as f:
        json.dump(persona, f, indent=2)
    print(f"[persona] Saved → {PERSONA_PATH}")
    print(json.dumps(persona, indent=2))
    return persona


if __name__ == "__main__":
    run_and_save()
