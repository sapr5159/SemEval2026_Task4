import json
from pathlib import Path

print("HELLO FROM DATASET_UTILS (JSONL VERSION)")  # sanity check


def load_track_b_texts(path: str | Path):
    """
    Load story texts from a JSONL file:
    Each line is like:
    {"text": "story ..."}
    """
    path = Path(path)
    texts = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                texts.append(obj.get("text", "").strip())

    print("Number of stories:", len(texts))
    if texts:
        print("\nFirst story preview:\n")
        print(texts[0][:400])

    return texts


if __name__ == "__main__":
    load_track_b_texts("data/dev_track_b.jsonl")