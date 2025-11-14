import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from dataset_utils import load_track_b_texts


def main():
    parser = argparse.ArgumentParser(
        description="Export Track B embeddings for a JSONL stories file."
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to input JSONL file (Track B-style, with 'text' field).",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Path to output JSONL file with embeddings.",
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

    # 1. Load texts
    print(f"Loading texts from: {args.input_jsonl}")
    texts = load_track_b_texts(args.input_jsonl)

    # 2. Load model
    print(f"\nLoading model: {args.model_name_or_path}")
    model = SentenceTransformer(args.model_name_or_path)

    # 3. Encode
    print("\nEncoding stories into embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("\nEmbedding shape:", embeddings.shape)

    # 4. Save as JSONL: one line per story
    output_path = Path(args.output_jsonl)
    print(f"\nWriting embeddings to: {output_path.resolve()}")

    with output_path.open("w", encoding="utf-8") as f:
        for idx, vec in enumerate(embeddings):
            row = {
                "id": idx,              # you can change this to story ID later if needed
                "embedding": vec.tolist(),
            }
            f.write(json.dumps(row) + "\n")

    print("\nDone. JSONL embeddings written successfully.")


if __name__ == "__main__":
    main()