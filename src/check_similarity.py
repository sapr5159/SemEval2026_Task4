import numpy as np
from sentence_transformers import util
from dataset_utils import load_track_b_texts


def main():
    # Load texts
    texts = load_track_b_texts("data/dev_track_b.jsonl")

    # Load embeddings
    embeddings = np.load("dev_track_b_baseline_embeddings.npy")

    print("\nEmbeddings loaded:", embeddings.shape)

    # Pick a story to test
    index = 0  # change this to try different ones
    query_text = texts[index]
    query_vec = embeddings[index]

    print("\n=== QUERY STORY ===")
    print(query_text)

    # Compute cosine similarity with all others
    cos_scores = util.cos_sim(query_vec, embeddings)[0].cpu().tolist()

    # Sort by similarity (highest first)
    sorted_indices = np.argsort(cos_scores)[::-1]

    print("\n=== MOST SIMILAR STORIES ===")
    for i in sorted_indices[1:6]:  # skip itself (i=0)
        print(f"\nScore: {cos_scores[i]:.4f}")
        print(texts[i][:400], "...")


if __name__ == "__main__":
    main()