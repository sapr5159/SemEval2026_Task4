# SemEval‑2026 Task 4 — Narrative Similarity (Tracks A & B)

> **Team**: **Sathish Kumar Prabaharan**, **Rishekesan Senthilkumar Vanathi**, **Mrunal Bhosale**  

Primary focus: **Track B** (one vector per story, cosine decision). We also provide a clean **Track A** baseline using a prompt‑engineered LLM judge. The repo is designed to be **training‑free, reproducible, and CPU‑friendly** after batch embedding.

---

## Results
| Track | Model / Extras                      | Dataset                  | Acc. |
|:----:|-------------------------------------|--------------------------|:---:|
| B    | BGE (base)                           | 200 dev                  | 0.640 |
| B    | E5 (base)                            | 200 dev                  | 0.650 |
| B    | GTE (base)                           | 200 dev                  | 0.595 |
| B    | **E5‑Large + Fusion + ABTT**         | **200 dev**              | **0.660** |
| B    | E5‑Large + Fusion + ABTT (+100 synth)| 200 dev + **100 synth*** | 0.770 |
| A    | Llama‑3.1‑8B‑Instant                  | 200 dev                  | 0.605 |
| A    | **Llama‑3.3‑70B‑Versatile**          | **200 dev**              | **0.745** |

\* *Dev‑only augmentation. Synthetic rows are **never** used in official 200‑item scoring.*

---

## Table of Contents
- [Environment](#environment)
- [Data](#data)
- [Method (Track B)](#method-track-b)
- [Quickstart (Track B)](#quickstart-track-b)
- [LLM Judge (Track A)](#llm-judge-track-a)
- [Exported Artifacts](#exported-artifacts)
- [Results & Notes](#results--notes)
- [CLI Flags](#cli-flags)
- [Troubleshooting](#troubleshooting)
- [Repo Layout](#repo-layout)
- [Citation](#citation)
- [License](#license)

---

## Environment

```bash
# create & activate venv (PowerShell shown; on bash: source .venv/bin/activate)
python -m venv .venv
.\.venv\Scripts\activate

pip install -U pip
pip install sentence-transformers numpy scipy scikit-learn tqdm
# (optional) for Track A judge
pip install groq google-generativeai
```

- Recommended: **Python 3.10+**.  
- If you see `TypeError: 'type' object is not subscriptable` for type hints
  like `tuple[np.ndarray, List[str]]`, use Python ≥3.9 or change to `typing.Tuple`.

**API keys (Track A only)**

```powershell
setx GROQ_API_KEY "your_groq_key_here"
# (optional if you try Gemini embeddings)
setx GOOGLE_API_KEY "your_google_key_here"
# open a NEW terminal after setx
```

---

## Data: 
- Get the development data from this website: https://narrative-similarity-task.github.io/data/
- **Track A** – `data/dev_track_a.jsonl` (200 rows)  
  Fields: `anchor_text`, `text_a`, `text_b`, and either `"label":"A"|"B"` or boolean `text_a_is_closer`.
- **Track B** – `data/dev_track_b.jsonl` (479 rows)  
  Each row: a single story text (no labels).

> We use the 200 dev triplets as a **proxy** to evaluate Track‑B encoders by comparing cosine decisions against the provided A/B signal.

---

## Method (Track B)

**Goal**: produce **one vector per story** and decide with cosine — no cross‑encoders, no pairwise attention.

1. **Base encoders**: E5 / BGE / GTE via `sentence-transformers`.  
   For E5/BGE, we apply instruction prefixes (`query:` for anchors, `passage:` for candidates); always **ℓ2‑normalize**.
2. **Chunk–Mean Pooling**: split long stories (~**1200** chars) with **200** char overlap, embed each chunk, mean‑pool, then re‑normalize.
3. **Aspect Fusion**: three views — **raw**, **verbs‑only**, **ending (last sentences)**.  
   Combine with weights `(0.5, 0.25, 0.25)` and re‑normalize.
4. **ABTT**: subtract mean & remove top‑**1** PC, fit on the **current** vector pool (anchors+candidates), then re‑normalize.
5. **Decision**: A if `cos(vx, vA) ≥ cos(vx, vB)` (ties → A). We log the **margin** `Δ = cos_A − cos_B` for confidence.

All steps are deterministic and Track‑B compliant.

---

## Quickstart (Track B)

### Baselines
```powershell
# BGE
python track_b_final.py `
  --devA data/dev_track_a.jsonl --devB data/dev_track_b.jsonl `
  --model BAAI/bge-base-en-v1.5 --prefix bge `
  --chunk-chars 1200 --chunk-overlap 200 --batch 32 --reembed-all

# E5
python track_b_final.py `
  --devA data/dev_track_a.jsonl --devB data/dev_track_b.jsonl `
  --model intfloat/e5-large-v2 --prefix e5 `
  --chunk-chars 1200 --chunk-overlap 200 --batch 32 --reembed-all
```

### Fusion + ABTT
```powershell
python track_b_final_boost.py `
  --devA data/dev_track_a.jsonl --devB data/dev_track_b.jsonl `
  --model intfloat/e5-large-v2 --prefix e5 `
  --chunk-chars 1200 --chunk-overlap 200 --batch 32 `
  --fusion-weights 0.5 0.25 0.25 --abtt-k 1 --reembed-all
```

*(Dev‑only)* add **+100** synthetic triplets for analysis:
```powershell
python track_b_final_boost.py ... --add-synthetic 100
```

Expected: `used=200, accuracy≈0.660` (and ≈0.770 for 200+100 synth; keep synth **out** of official 200).

---

## LLM Judge (Track A)

```powershell
# larger model (higher accuracy)
python track_a_groq.py --devA data/dev_track_a.jsonl ^
  --model llama-3.3-70b-versatile --temperature 0.0

# faster model
python track_a_groq.py --devA data/dev_track_a.jsonl ^
  --model llama-3.1-8b-instant --temperature 0.0
```

**Prompt template (system/user):**
```
System:
You are a judge for narrative similarity. Compare the anchor to A and B on:
(1) plot/theme, (2) actions/events, (3) outcome/ending, (4) setting/characters).
Return ONLY strict JSON: {"label":"A"} or {"label":"B"} (no text).

User:
{"anchor":"...", "A":"...", "B":"..."}
```
- Strict JSON parsing; back‑off to minimal "A"/"B" match; default to **A** if undecidable.
- Use `temperature=0.0` for determinism.

---

## Exported Artifacts

Scripts write:
```
outputs/
  track_b.npy          # [N, D] float32, L2‑normalized vectors (one per story)
  track_b_ids.jsonl    # aligned IDs (one per line)
```
Example:
```python
import json, numpy as np
V = np.load("outputs/track_b.npy")
ids = [json.loads(l)["id"] for l in open("outputs/track_b_ids.jsonl","r",encoding="utf-8")]
def cos(a,b): return float(np.dot(a,b))
print(ids[0], ids[1], cos(V[0], V[1]))
```
We export **after** chunking+fusion and **before** ABTT so downstream users can fit ABTT on their own pool.

---

## Results & Notes

- For 200 items, 95% normal CI is wide (~±0.066 near 0.66). Treat small deltas cautiously.
- **Margins** correlate with correctness; using Δ for abstention (drop lowest 10–20%) improves stability at reduced coverage.
- Common errors: negation/counterfactuals, entity‑role swaps, long‑range causal links, stylistic confounds — areas where the Track‑A judge helps.

---

## CLI Flags

**Common**  
`--devA`, `--devB` (paths) · `--model` (e.g., `intfloat/e5-large-v2`) · `--prefix {e5,bge,none}` · `--batch` · `--chunk-chars` (default 1200) · `--chunk-overlap` (default 200) · `--reembed-all`

**Fusion/ABTT (boost script)**  
`--fusion-weights <raw> <verb> <end>` (default `0.5 0.25 0.25`) · `--abtt-k` (default `1`) · `--add-synthetic N` (dev‑only)  

**Track A**  
`--model {llama-3.1-8b-instant, llama-3.3-70b-versatile}` · `--temperature` (default `0.0`)  

---

## Troubleshooting

- **Typing error** (`'type' object is not subscriptable`) → use Python ≥3.9 or `typing.Tuple[...]`.
- **Gemini embeddings**: some SDKs expose `genai.embed_content`, others `genai.embeddings.embed_content`. If `AttributeError`, try:
  ```python
  import google.generativeai as genai, numpy as np
  genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
  r = genai.embed_content(model="text-embedding-004", content=text)
  vec = np.array(r["embedding"], dtype=np.float32)
  ```
- **PowerShell env vars**: after `setx`, open a **new** terminal.
- **Vector norms**: ensure L2 re‑normalization after chunk mean, fusion, and ABTT (‖v‖₂≈1).

---

## Repo Layout

```
.
├─ data/
│  ├─ dev_track_a.jsonl
│  └─ dev_track_b.jsonl
├─ track_b_final.py
├─ track_b_final_boost.py
├─ track_b.py
├─ track_a_groq.py
├─ outputs/
│  ├─ track_b.npy
│  └─ track_b_ids.jsonl
└─ README.md
```

---

## Citation

If you build on this work:

```
@misc{sem2026-track4-aspectfusion,
  title  = {SemEval-2026 Task 4: Track-B Per-Story Embeddings via Aspect Fusion, Chunk–Mean Pooling and ABTT, with a Prompt-Engineered LLM Baseline for Track A},
  author = {Prabaharan, Sathish Kumar and Vanathi, Rishekesan Senthilkumar and Bhosale, Mrunal},
  year   = {2026},
  howpublished = {\url{https://github.com/sapr5159/SemEval2026_Task4}}
}
```

---

## License

- Uses open encoders from `sentence-transformers` (E5/BGE/GTE).
- LLM judge uses Groq‑hosted Llama models.
- Data: SemEval‑2026 Task 4 dev files.
- Public code, no PII. Add a `LICENSE` file if not present.
