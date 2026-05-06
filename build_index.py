"""
build_index.py
--------------
Run this ONCE to preprocess everything and save to disk.
After this, the chatbot loads from disk — no reprocessing needed.

Steps:
1. Parse all messages
2. Detect topic checkpoints + build 100-msg checkpoints
3. Build FAISS index
4. Extract persona

Usage:
    python build_index.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "core"))

from parser import load_messages
from topic_detector import run_and_save as build_checkpoints, load_checkpoints, CHECKPOINT_PATH
from rag_engine import build_index, INDEX_PATH
from persona_extractor import run_and_save as build_persona, PERSONA_PATH


def main():
    t0 = time.time()
    print("=" * 60)
    print("BUILDING RAG INDEX — run this once before starting chatbot")
    print("=" * 60)

    # Step 1: Parse
    print("\n[1/4] Parsing messages...")
    messages = load_messages()
    print(f"      {len(messages)} messages loaded")

    # Step 2: Topic detection
    if CHECKPOINT_PATH.exists():
        print("\n[2/4] Topic checkpoints already exist, skipping...")
        checkpoints = load_checkpoints()
    else:
        print("\n[2/4] Detecting topics + building 100-msg checkpoints...")
        checkpoints = build_checkpoints(messages)

    print(f"      Topics detected: {len(checkpoints['topic_checkpoints'])}")
    print(f"      100-msg checkpoints: {len(checkpoints['message_checkpoints'])}")

    # Step 3: FAISS index
    if INDEX_PATH.exists():
        print("\n[3/4] FAISS index already exists, skipping...")
    else:
        print("\n[3/4] Building FAISS index...")
        build_index(messages, checkpoints)

    # Step 4: Persona
    if PERSONA_PATH.exists():
        print("\n[4/4] Persona already exists, skipping...")
    else:
        print("\n[4/4] Extracting user persona...")
        build_persona()

    elapsed = round(time.time() - t0, 1)
    print(f"\n{'='*60}")
    print(f"All done in {elapsed}s. You can now run: streamlit run chatbot/app.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
