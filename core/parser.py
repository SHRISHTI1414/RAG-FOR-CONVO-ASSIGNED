"""
parser.py
---------
Reads conversations.csv and flattens everything into a single
chronological list of messages. Each row in the CSV is one conversation
(treated as one "day"). Messages inside each conversation are ordered
as-is (they're already sequential).

Output format per message:
{
    "global_idx": int,       # position in the full flat stream
    "conv_idx": int,         # which conversation (row) this came from
    "turn_idx": int,         # position within that conversation
    "speaker": "User 1" | "User 2",
    "text": str
}
"""

import re
import pandas as pd
from pathlib import Path


DATA_PATH = Path(__file__).parent.parent / "data" / "conversations.csv"
MSG_PATTERN = re.compile(r'^(User [12]):\s*(.+)$')


def load_messages(csv_path: Path = DATA_PATH) -> list[dict]:
    df = pd.read_csv(csv_path, header=0)
    df.columns = ["conversation"]

    flat = []
    global_idx = 0

    for conv_idx, row in df.iterrows():
        raw_lines = row["conversation"].split("\n")
        turn_idx = 0

        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            m = MSG_PATTERN.match(line)
            if not m:
                continue

            speaker = m.group(1).strip()
            text = m.group(2).strip()

            flat.append({
                "global_idx": global_idx,
                "conv_idx": int(conv_idx),
                "turn_idx": turn_idx,
                "speaker": speaker,
                "text": text,
            })
            global_idx += 1
            turn_idx += 1

    return flat


if __name__ == "__main__":
    msgs = load_messages()
    print(f"Total messages loaded: {len(msgs)}")
    print("First 5:")
    for m in msgs[:5]:
        print(f"  [{m['global_idx']}] conv={m['conv_idx']} | {m['speaker']}: {m['text'][:60]}")
    print("Last 3:")
    for m in msgs[-3:]:
        print(f"  [{m['global_idx']}] conv={m['conv_idx']} | {m['speaker']}: {m['text'][:60]}")
