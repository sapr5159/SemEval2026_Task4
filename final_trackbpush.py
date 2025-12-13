# %% [1] IMPORTS
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


# %% [2] CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[Config] Device:", DEVICE)

# Base models
E5_NAME = "intfloat/e5-large-v2"
E5_PROMPT = "query: "  # required prefix for E5

BGE_NAME = "BAAI/bge-large-en-v1.5"
BGE_PROMPT = "Represent this sentence for searching relevant passages: "

# Paths
BASE_DIR = Path("data")
TRACK_B_STORIES_PATH = BASE_DIR / "dev_track_b.jsonl"
EVALUATION_DATA_PATH = BASE_DIR / "dev_track_a.jsonl"

OUT_JSON = Path("track_b.jsonl")
OUT_NPY = Path("track_b.npy")
OUT_ZIP = Path("codabench_track_b.zip")

# Metric-learning hyperparams (you can tweak these)
D_OUT = 256          # final embedding dim (from 2048 -> 256)
LR = 3e-3            # learning rate
WEIGHT_DECAY = 1e-3  # L2 regularization
EPOCHS = 10     # number of epochs

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"[Config] E5 model:  {E5_NAME}")
print(f"[Config] BGE model: {BGE_NAME}")
print(f"[Config] Metric dim: {D_OUT}, LR={LR}, WD={WEIGHT_DECAY}, EPOCHS={EPOCHS}")


# %% [3] LOAD ALL UNIQUE STORIES
def read_unique_stories(*paths):
    """
    Reads all jsonl files and returns a list of unique story texts,
    preserving first-seen order.
    """
    unique_texts = []
    seen = set()
    text_cols = ["text", "story", "content", "anchor_text", "text_a", "text_b"]

    for path in paths:
        print(f"[IO] Reading stories from {path}...")
        df = pd.read_json(path, lines=True)

        for col in text_cols:
            if col in df.columns:
                for txt in df[col].dropna():
                    if txt not in seen:
                        seen.add(txt)
                        unique_texts.append(txt)

    print(f"[IO] Found {len(unique_texts)} unique stories in total.")
    return unique_texts


all_stories = read_unique_stories(TRACK_B_STORIES_PATH, EVALUATION_DATA_PATH)
assert len(all_stories) > 0, "No stories found!"

# Map story text -> index
story_to_idx = {t: i for i, t in enumerate(all_stories)}


# %% [4] LOAD MODELS
def load_model(name):
    print(f"[Model] Loading {name} ...")
    model = SentenceTransformer(name, device=DEVICE)
    dim = model.get_sentence_embedding_dimension()
    print(f"[Model] {name} loaded (dim={dim})")
    return model, dim


e5_model, e5_dim = load_model(E5_NAME)
bge_model, bge_dim = load_model(BGE_NAME)


# %% [5] ENCODING HELPERS
@torch.no_grad()
def encode_with(model, prompt, texts):
    """
    Encode texts with a given SentenceTransformer and prompt prefix.
    Returns L2-normalized np.float32 embeddings.
    """
    texts_with_prompt = [prompt + t for t in texts]
    embs = model.encode(
        texts_with_prompt,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    return embs


print("\n[Encode] Encoding all unique stories with E5...")
e5_embs = encode_with(e5_model, E5_PROMPT, all_stories)
print("[Encode] E5 embeddings shape:", e5_embs.shape)

print("\n[Encode] Encoding all unique stories with BGE...")
bge_embs = encode_with(bge_model, BGE_PROMPT, all_stories)
print("[Encode] BGE embeddings shape:", bge_embs.shape)

# Concatenate base features: [E5 | BGE]
base_embs = np.concatenate([e5_embs, bge_embs], axis=1)
n_stories, d_in = base_embs.shape
print("[Encode] Concatenated base embeddings shape:", base_embs.shape)


# %% [6] LOAD DEV TRIPLETS
dev_df = pd.read_json(EVALUATION_DATA_PATH, lines=True)
print(f"\n[Data] dev_track_a triples: {len(dev_df)}")
print("[Data] Label balance (text_a_is_closer):", dev_df["text_a_is_closer"].mean())


# Build index tensors for triples
anchor_idx = []
a_idx = []
b_idx = []
labels = []

for row in dev_df.itertuples(index=False):
    anchor_idx.append(story_to_idx[row.anchor_text])
    a_idx.append(story_to_idx[row.text_a])
    b_idx.append(story_to_idx[row.text_b])
    # y = +1 if text_a is closer, -1 otherwise
    y = 1.0 if row.text_a_is_closer else -1.0
    labels.append(y)

anchor_idx = torch.tensor(anchor_idx, dtype=torch.long, device=DEVICE)
a_idx = torch.tensor(a_idx, dtype=torch.long, device=DEVICE)
b_idx = torch.tensor(b_idx, dtype=torch.long, device=DEVICE)
labels = torch.tensor(labels, dtype=torch.float32, device=DEVICE)

print("[Data] anchor_idx shape:", anchor_idx.shape)
print("[Data] labels example:", labels[:10])


# %% [7] BASELINE ACCURACY (no metric learning, just base_embs)
def eval_accuracy_from_embs(embs_np):
    """
    Compute dev accuracy if we just use cosine similarity on embs_np.
    """
    embs = torch.tensor(embs_np, dtype=torch.float32, device=DEVICE)
    embs = F.normalize(embs, dim=1)  # ensure cosine

    with torch.no_grad():
        anc = embs[anchor_idx]
        a = embs[a_idx]
        b = embs[b_idx]

        sim_a = (anc * a).sum(dim=1)
        sim_b = (anc * b).sum(dim=1)

        preds = (sim_a > sim_b).float()
        # labels: +1 means A closer, -1 means B closer
        labels01 = (labels > 0).float()
        acc = (preds == labels01).float().mean().item()
    return acc


baseline_acc = eval_accuracy_from_embs(base_embs)
print(f"\n[Baseline] Accuracy with base concatenated embeddings: {baseline_acc:.4f}")


# %% [8] METRIC LEARNING: LEARN L SUCH THAT
# score = <Lx_anchor, Lx_A> - <Lx_anchor, Lx_B> is positive when A is closer.
X = torch.tensor(base_embs, dtype=torch.float32, device=DEVICE)  # [N_stories, d_in]

# Projection matrix L: [D_OUT, d_in], final embedding = X @ L.T
L = torch.nn.Parameter(0.01 * torch.randn(D_OUT, d_in, device=DEVICE))

optimizer = torch.optim.AdamW([L], lr=LR, weight_decay=WEIGHT_DECAY)


def metric_learning_step():
    optimizer.zero_grad()

    # current projected embeddings
    Z = X @ L.t()  # [N_stories, D_OUT]

    anc = Z[anchor_idx]  # [N_triples, D_OUT]
    a = Z[a_idx]
    b = Z[b_idx]

    sim_a = (anc * a).sum(dim=1)
    sim_b = (anc * b).sum(dim=1)

    # score > 0 when A is closer; labels in {+1, -1}
    score = sim_a - sim_b

    # logistic-style loss: softplus(-y * score)
    loss = F.softplus(-labels * score).mean()

    loss.backward()
    optimizer.step()
    return loss.item()


def eval_accuracy_current_L():
    with torch.no_grad():
        Z = X @ L.t()
        anc = Z[anchor_idx]
        a = Z[a_idx]
        b = Z[b_idx]
        sim_a = (anc * a).sum(dim=1)
        sim_b = (anc * b).sum(dim=1)
        preds = (sim_a > sim_b).float()
        labels01 = (labels > 0).float()
        acc = (preds == labels01).float().mean().item()
    return acc


print("\n[Training] Starting metric learning...")
best_acc = eval_accuracy_current_L()
best_state = L.detach().clone()
print(f"[Training] Initial accuracy with random L: {best_acc:.4f}")

for epoch in range(1, EPOCHS + 1):
    loss = metric_learning_step()

    if epoch % 50 == 0 or epoch == 1:
        acc = eval_accuracy_current_L()
        print(f"  Epoch {epoch:4d} | loss={loss:.4f} | dev_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = L.detach().clone()

# Restore best L (by dev accuracy)
L.data = best_state
final_dev_acc = eval_accuracy_current_L()
print("\n[Training] Best dev accuracy during training:", best_acc)
print("[Training] Final dev accuracy with best L:", final_dev_acc)


# %% [9] BUILD FINAL EMBEDDINGS FOR ALL STORIES
with torch.no_grad():
    Z_all = (X @ L.t()).cpu().numpy().astype("float32")
print("[Embeddings] Final learned embeddings shape:", Z_all.shape)

# Rebuild lookup story -> embedding
embedding_lookup = {text: emb for text, emb in zip(all_stories, Z_all)}


# %% [10] SANITY CHECK: EVAL AGAIN USING LOOKUP + DOT PRODUCT
def calculate_prediction_score_from_lookup(df, lookup):
    sims_a = []
    sims_b = []
    for row in df.itertuples(index=False):
        anc = lookup[row.anchor_text]
        A = lookup[row.text_a]
        B = lookup[row.text_b]
        sim_a = float(np.dot(anc, A))
        sim_b = float(np.dot(anc, B))
        sims_a.append(sim_a)
        sims_b.append(sim_b)
    sims_a = np.array(sims_a)
    sims_b = np.array(sims_b)
    preds = sims_a > sims_b
    labels_np = dev_df["text_a_is_closer"].values
    acc = (preds == labels_np).mean()
    return acc


check_acc = calculate_prediction_score_from_lookup(dev_df, embedding_lookup)
print("\n===== LOCAL PERFORMANCE ESTIMATE =====")
print(f"Baseline (no metric learning): {baseline_acc:.4f}")
print(f"Final with learned metric L : {check_acc:.4f}")
print("======================================")
print("(This score is an estimate of your performance on the official leaderboard)")


# %% [11] SAVE FOR SUBMISSION (TRACK B)
print("\n[Export] Preparing submission files...")

track_b_df = pd.read_json(TRACK_B_STORIES_PATH, lines=True)

if "text" not in track_b_df.columns:
    raise ValueError("Expected a 'text' column in dev_track_b.jsonl")

submission_stories = track_b_df["text"].tolist()
print(f"[Export] Using column 'text' with {len(submission_stories)} stories.")

submission_embeddings = np.stack([embedding_lookup[txt] for txt in submission_stories])
print(f"[Export] Submission embeddings shape: {submission_embeddings.shape}")

# Save .npy
np.save(OUT_NPY, submission_embeddings)

# Save .jsonl
with open(OUT_JSON, "w", encoding="utf-8") as f:
    for emb in submission_embeddings.tolist():
        f.write(json.dumps({"embedding": emb}) + "\n")

# Zip for Codabench
with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write(OUT_JSON, arcname="track_b.jsonl")
    z.write(OUT_NPY, arcname="track_b.npy")

print(f"[Export] Saved to:\n  • {OUT_JSON}\n  • {OUT_NPY}\n  • {OUT_ZIP}")
print("\nDone ✅")
