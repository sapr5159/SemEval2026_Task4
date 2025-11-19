import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, util


def load_triplets(jsonl_path: str):
    """Load (anchor, opt1, opt2, label) from a JSONL triples file."""
    path = Path(jsonl_path)
    triplets = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            # 🔁 CHANGE THESE KEYS IF YOUR FILE USES DIFFERENT NAMES
            anchor = obj["anchor"]
            opt1 = obj["option1"]
            opt2 = obj["option2"]
            label = obj["label"]  # e.g. 1 if opt1 is correct, 2 if opt2 is correct

            triplets.append((anchor, opt1, opt2, label))

    return triplets


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a sentence embedding model on anchor-choice triplets."
    )
    parser.add_argument(
        "--triplets_jsonl",
        type=str,
        required=True,
        help="Path to JSONL file with anchor/option1/option2/label.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="models/tsdae_story_encoder",
        help="SentenceTransformer model name or local path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding.",
    )

    args = parser.parse_args()

    # 1. Load triplets
    triplets = load_triplets(args.triplets_jsonl)
    print(f"Loaded {len(triplets)} triplets from {args.triplets_jsonl}")

    # 2. Load model
    print(f"\nLoading model: {args.model_name_or_path}")
    model = SentenceTransformer(args.model_name_or_path)

    # 3. Encode all texts (anchor + opt1 + opt2) in one big batch
    anchors = [t[0] for t in triplets]
    opts1 = [t[1] for t in triplets]
    opts2 = [t[2] for t in triplets]

    print("\nEncoding anchors...")
    emb_anchors = model.encode(
        anchors,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print("\nEncoding option1...")
    emb_opts1 = model.encode(
        opts1,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print("\nEncoding option2...")
    emb_opts2 = model.encode(
        opts2,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # 4. Compute accuracy
    correct = 0
    margins = []

    for i, (_, _, _, label) in enumerate(triplets):
        a = emb_anchors[i]
        b1 = emb_opts1[i]
        b2 = emb_opts2[i]

        sim1 = float(util.cos_sim(a, b1))
        sim2 = float(util.cos_sim(a, b2))

        # predicted: 1 if opt1 more similar, 2 if opt2 more similar
        pred = 1 if sim1 >= sim2 else 2
        if pred == label:
            correct += 1

        margins.append(abs(sim1 - sim2))

    accuracy = correct / len(triplets)
    avg_margin = float(np.mean(margins))

    print(f"\nAccuracy: {accuracy:.4f}  ({correct} / {len(triplets)})")
    print(f"Average similarity margin |sim1 - sim2|: {avg_margin:.4f}")


if __name__ == "__main__":
    main()