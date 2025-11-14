from pathlib import Path

from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    datasets,
    losses,
)

from dataset_utils import load_track_b_texts


def main():
    # 1. Load texts
    texts = load_track_b_texts("data/dev_track_b.jsonl")

    # 2. Base model
    base_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"\nLoading base model for TSDAE: {base_model_name}")
    model = SentenceTransformer(base_model_name)

    # 3. Build TSDAE dataset (denoising autoencoder)
    print("\nBuilding TSDAE dataset...")
    train_dataset = datasets.DenoisingAutoEncoderDataset(texts)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 4. Loss: Denoising AutoEncoder
    train_loss = losses.DenoisingAutoEncoderLoss(
        model,
        decoder_name_or_path=base_model_name,
    )

    # 5. Train
    output_dir = Path("models/tsdae_story_encoder")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting TSDAE training (1 epoch)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,                          # start small; we can increase later
        optimizer_params={"lr": 2e-5},
        output_path=str(output_dir),
        show_progress_bar=True,
    )

    print(f"\nTSDAE fine-tuned model saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()