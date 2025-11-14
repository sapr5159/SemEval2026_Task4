from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from dataset_utils import load_track_b_texts


def main():
    # 1. Load all story texts
    texts = load_track_b_texts("data/dev_track_b.jsonl")

    # 2. Load a pre-trained sentence embedding model

    model_name = "models/tsdae_story_encoder"
    print(f"\nLoading fine-tuned model: {model_name}")
    model = SentenceTransformer(model_name)

    # 3. Encode texts into embeddings
    print("\nEncoding stories into embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # nice for cosine similarity
    )

    print("\nEmbedding shape:", embeddings.shape)  # (num_stories, dim)

    # 4. Save embeddings as a .npy file
    output_path = Path("dev_track_b_tsdae_embeddings.npy")
    np.save(output_path, embeddings)
    print(f"\nSaved embeddings to: {output_path.resolve()}")


if __name__ == "__main__":
    main()