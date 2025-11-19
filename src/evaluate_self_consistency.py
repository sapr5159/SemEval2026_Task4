import argparse
import random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, util

from dataset_utils import load_track_b_texts


def build_eval_triplets(texts, num_anchors: int = 200, seed: int = 42):
    """
    Build evaluation triplets (anchor, positive_view, negative_story).

    - anchor: full story text
    - positive_view: truncated version of the SAME story
    - negative_story: a DIFFERENT random story
    """
    random.seed(seed)
    n = len(texts)
    indices = list(range(n))
    random.shuffle(indices)
    indices = indices[: min(num_anchors, n)]

    triplets = []

    for i in indices:
        anchor = texts[i]
        # Positive: truncated version of same story (first half)
        half = max(50, len(anchor) // 2)  # keep at least 50 chars
        positive = anchor[:half]

        # Negative: random different story
        j = random.randrange(n)
        while j == i:
            j = random.randrange(n)
        negative = texts[j]

        triplets.append((anchor, positive, negative))

    return triplets


def evaluate_self_consistency(model_name_or_path: str, num_anchors: int = 200):
    print(f"\nLoading texts from Track B dev file...")
    texts = load_track_b_texts("data/dev_track_b.jsonl")
    print(f"Total stories: {len(texts)}")

    print(f"\nBuilding {num_anchors} self-consistency triplets...")
    triplets = build_eval_triplets(texts, num_anchors=num_anchors)
    print(f"Built {len(triplets)} triplets.")

    print(f"\nLoading model: {model_name_or_path}")
    model = SentenceTransformer(model_name_or_path)

    # Collect all unique texts we need to embed
    unique_texts = {}
    for anchor, pos, neg in triplets:
        unique_texts.setdefault(anchor, None)
        unique_texts.setdefault(pos, None)
        unique_texts.setdefault(neg, None)

    all_text_list = list(unique_texts.keys())
    print(f"\nEncoding {len(all_text_list)} unique snippets...")
    embeddings = model.encode(
        all_text_list,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Map text -> embedding index
    text2idx = {text: i for i, text in enumerate(all_text_list)}

    correct = 0
    pos_sims = []
    neg_sims = []

    for anchor, pos, neg in triplets:
        a = embeddings[text2idx[anchor]]
        p = embeddings[text2idx[pos]]
        n = embeddings[text2idx[neg]]

        sim_pos = float(util.cos_sim(a, p))
        sim_neg = float(util.cos_sim(a, n))

        pos_sims.append(sim_pos)
        neg_sims.append(sim_neg)

        if sim_pos > sim_neg:
            correct += 1

    accuracy = correct / len(triplets)
    pos_mean = float(np.mean(pos_sims))
    neg_mean = float(np.mean(neg_sims))

    print("\n=== Self-consistency evaluation ===")
    print(f"Model: {model_name_or_path}")
    print(f"Anchors evaluated: {len(triplets)}")
    print(f"Accuracy (sim(anchor,pos) > sim(anchor,neg)): {accuracy:.4f}")
    print(f"Mean positive similarity: {pos_mean:.4f}")
    print(f"Mean negative similarity: {neg_mean:.4f}")
    print(f"Mean margin (pos - neg): {pos_mean - neg_mean:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Self-consistency evaluation for Track B embeddings."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="models/tsdae_story_encoder",
        help="SentenceTransformer model name or local path.",
    )
    parser.add_argument(
        "--num_anchors",
        type=int,
        default=200,
        help="Number of anchor stories to use for evaluation.",
    )

    args = parser.parse_args()
    evaluate_self_consistency(args.model_name_or_path, args.num_anchors)


if __name__ == "__main__":
    main()