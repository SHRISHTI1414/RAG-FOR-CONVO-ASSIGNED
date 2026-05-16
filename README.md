# Conversation RAG Chatbot

A fully local RAG system built over 11,000 conversations (~191,000 messages).  
**No external LLM APIs. No LangChain. Everything runs on your machine.**

---

## Quick Start

```bash
git clone <your-repo-url>
cd ragbot

pip install -r requirements.txt

# Put the dataset here
cp /path/to/conversations.csv data/

# One-time preprocessing (builds all indexes, ~10–15 min on full dataset)
python build_index.py

# Launch chatbot
streamlit run chatbot/app.py
```

---

## Project Structure

```
ragbot/
├── core/
│   ├── parser.py            # CSV → flat chronological message stream
│   ├── topic_detector.py    # sliding-window cosine similarity → topic checkpoints
│   ├── persona_extractor.py # rule-based trait extraction with evidence
│   ├── rag_engine.py        # FAISS index + ranked retrieval
│   └── generator.py         # flan-t5-base answer generation + source citations
├── chatbot/
│   └── app.py               # Streamlit UI
├── data/                    # conversations.csv + generated .pkl files
├── build_index.py           # one-time preprocessing script
└── requirements.txt
```

---

## Part 1: RAG System with Checkpoints

### How Topic Detection Works

All 11,000 conversation rows are flattened into a single chronological message stream (191,578 messages total). Row index = day index, so row 0 message 1 → row 0 message 17 → row 1 message 1 etc.

**Algorithm:**

```
For each position i in the stream (step = WINDOW_SIZE / 2):

    window_A = messages[i     : i + WINDOW_SIZE]
    window_B = messages[i + WINDOW_SIZE : i + WINDOW_SIZE * 2]

    emb_A = embed(join(window_A.text))
    emb_B = embed(join(window_B.text))

    similarity = cosine_similarity(emb_A, emb_B)   # float in [-1, 1]

    if similarity < 0.35:
        → mark a topic boundary at position i + WINDOW_SIZE
```

**Why this works:**  
Messages in the same topic share vocabulary and semantic meaning → high cosine similarity. When the subject shifts (e.g., cooking → travel), the two windows diverge in embedding space → similarity drops sharply. The threshold 0.35 was chosen empirically: above it, differences are conversational variation within a topic; below it, the conversation has moved elsewhere.

**Parameters:**
| Parameter | Value | Reason |
|-----------|-------|--------|
| `WINDOW_SIZE` | 8 messages | Enough context to represent a topic without averaging out shifts |
| `THRESHOLD` | 0.35 | Empirically tuned — catches real shifts, ignores noise |
| `MIN_TOPIC_LEN` | 15 messages | Prevents micro-topics from a single off-topic line |
| Window step | 4 (50% overlap) | Smoother boundary detection, doesn't miss boundaries at step edges |

**Output per topic checkpoint:**
```json
{
  "topic_id": 12,
  "start_index": 264,
  "end_index": 311,
  "message_count": 48,
  "boundary_similarity": 0.082,
  "summary": "User 1 talks about hiking in national parks ... User 2 mentions Yellowstone ...",
  "key_phrases": ["national parks", "hiking trail", "camping gear", "yellowstone"],
  "messages": [ ... full message objects ... ]
}
```

### 100-Message Checkpoints

Every 100 messages in the flat stream gets a separate checkpoint. These are completely independent of topics — they're a coarse time-based summary useful when the query relates to a broad time window rather than a specific topic.

```json
{
  "checkpoint_id": 5,
  "start_index": 400,
  "end_index": 499,
  "message_count": 100,
  "summary": "Conversations cover cooking, pets, and work-life balance ..."
}
```

### Summaries

Both topic and 100-message checkpoints use **extractive summarisation**: the top-N sentences ranked by their summed TF-IDF score. No LLM required for summarisation — the highest-scoring sentences are the most content-rich ones in that segment.

---

## Part 2: User Persona

### How Persona Extraction Works

User 1's lines are extracted from all conversations (98,072 messages on the full dataset). Every trait is backed by actual message text — no guessing.

**Pattern banks** are defined for each trait category:

```python
HABIT_PATTERNS = {
    "reader": [
        r"\b(love|like|enjoy)\s+(to\s+)?read(ing)?\b",
        r"\b(reading|book|novel)\b",
    ],
    "late_sleeper": [
        r"\b(stay\s+up\s+late|night\s+owl|awake\s+at\s+\d+\s*am)\b",
    ],
    ...
}
```

For every message that matches a pattern, the actual message text is stored as evidence:

```json
{
  "habits": {
    "reader": {
      "confidence": "high",
      "hit_count": 4880,
      "evidence": [
        "Reading is a great way to relax. I love to cook too!",
        "Yes! I love reading to kids. It's the best part of my job!",
        "I just finished reading The Name of the Wind — amazing book."
      ]
    }
  }
}
```

**Confidence tiers:**
- `high` → ≥20 hits
- `medium` → 5–19 hits
- `low` → 2–4 hits
- Not included → <2 hits (noise threshold)

**Communication style** is derived from statistics, not patterns:
- `avg_message_length_words` — raw word count average
- `question_ratio` — fraction of messages containing `?`
- `exclamation_ratio` — fraction containing `!`
- `emoji_ratio` — fraction containing unicode emoji ranges
- `tone` — inferred from the above ratios

---

## Part 3: Chatbot Query Routing

```
user question
      │
      ├─ contains persona keywords? ─── Yes ──→ answer_persona_question()
      │   ("habits", "personality",             uses persona JSON directly
      │    "how do they talk", etc.)
      │
      └─ No ──→ RAGEngine.retrieve()
                    │
                    ├── embed query (same model as index)
                    ├── FAISS inner-product search (cosine sim on unit vecs)
                    ├── top-3 topic summaries  (ranked by similarity score)
                    └── top-3 message chunks   (ranked by similarity score)
                              │
                          build_context()
                              │
                          generator.answer()
                              │
                    structured response:
                    {
                      "answer": "...",
                      "sources": {
                        "topics": [{topic_id, start_index, end_index, score}],
                        "chunks": [{start_idx, end_idx, score}]
                      },
                      "model_used": "flan-t5-base"
                    }
```

Every answer in the UI shows **exactly which topic segments and message ranges** it was retrieved from, along with the cosine similarity score for each source.

---

## Stack

| Component | Library | Why |
|-----------|---------|-----|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | 384-dim, 22MB, fast CPU inference |
| Vector search | `faiss-cpu` (IndexFlatIP) | Exact cosine search, in-process, no server |
| Answer generation | `transformers` (flan-t5-base) | 250M params, CPU-friendly, instruction-tuned |
| Offline fallback | TF-IDF + TruncatedSVD (scikit-learn) | Works with no internet at all |
| UI | `streamlit` | Standard for ML demos |

**No external APIs used anywhere in the pipeline.**

---

## What's NOT Used (and Why)

- ❌ **LangChain / LlamaIndex** — these are wrappers. The retrieval logic here is written from scratch so the approach is transparent and explainable.
- ❌ **OpenAI / Anthropic / Gemini** — external LLMs would make the system dependent on an API key and network access.
- ❌ **ChromaDB / Pinecone / Weaviate** — FAISS runs in-process and is sufficient for this dataset size.


cat >> README.md << 'EOF'

---

## L2 — Adaptive Conversation Intelligence

### Part 1: Persona Drift Detector
- 1100 day profiles built, 1066 drift events detected
- Sliding window emotional scoring across 8 tone axes
- Trigger detection: work_stress, relationships, health, etc.
- Output: `data/drift_report.json`

### Part 2: Intent Classifier
- Model: TF-IDF + Logistic Regression (char n-grams)
- Size: 149KB (limit: 50MB)
- Avg latency: 0.2ms (limit: 200ms)
- F1 score: 0.766
- 5 classes: reminder, emotional_support, action_item, small_talk, unknown

### Part 3: Conflict Resolver
- Ranks chunks by: recency + emotional weight + relevance
- Detects contradictions using sentiment signature comparison
- Merges into single coherent answer with contradiction flags

### Part 4: System Design
- See `SYSTEM_DESIGN.md`
- On-device first, derived artefacts sync to cloud
- Raw conversations never leave device

### Run L2 Demo
```bash
streamlit run app_l2.py
```
EOF
git add README.md
git commit -m "update README with L2 documentation"
git push
