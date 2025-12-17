
import os, json, argparse, sys, time, math
from typing import Dict, List
import numpy as np
import pandas as pd

from groq import Groq

# ---------------------- Utilities ----------------------

def cos(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a/na, b/nb))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: str, rows: List[Dict]):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------------- Groq clients ----------------------

def get_client() -> Groq:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set. Use `export GROQ_API_KEY=...` (Windows: setx GROQ_API_KEY ...)")
    return Groq(api_key=key)

# ---------------------- Embedding scorer ----------------------

def embed_texts(client: Groq, texts: List[str], model: str = "nomic-embed-text-v1.5") -> List[List[float]]:
    # Single call; for very large batches you may chunk.
    resp = client.embeddings.create(model=model, input=texts, encoding_format="float")
    # preserve order
    return [d.embedding for d in resp.data]

def predict_with_embeddings(client: Groq, anchor: str, A: str, B: str, model: str) -> Dict:
    vecs = embed_texts(client, [anchor, A, B], model=model)
    sA = cos(vecs[0], vecs[1])
    sB = cos(vecs[0], vecs[2])
    choice = "A" if sA >= sB else "B"
    return {"choice": choice, "scores": {"emb_A": sA, "emb_B": sB}, "aspects": None}

# ---------------------- LLM judge (Structured JSON) ----------------------

SYSTEM_PROMPT = (
  "You are a careful judge for Narrative Similarity Task A. "
  "Given an Anchor and two candidates A and B, pick which is narratively closer. "
  "Ignore writing style, names, and settings. Consider abstract theme, course of action (event order), and outcomes."
)

def make_prompt(anchor: str, A: str, B: str) -> str:
    return (
f"""Anchor:
{anchor}

Candidate A:
{A}

Candidate B:
{B}

Return a strict JSON with:
- choice: "A" or "B"
- aspects: {{
    "abstract_theme": boolean,
    "course_of_action": boolean,
    "outcomes": boolean
  }}
- rationale: short one-liner justification.
"""
    )

JSON_SCHEMA = {
  "name": "narrative_task_a",
  "strict": True,
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "choice": {"type": "string", "enum": ["A","B"]},
      "aspects": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "abstract_theme": {"type": "boolean"},
          "course_of_action": {"type": "boolean"},
          "outcomes": {"type": "boolean"}
        },
        "required": ["abstract_theme","course_of_action","outcomes"]
      },
      "rationale": {"type": "string"}
    },
    "required": ["choice","aspects"]
  }
}

def predict_with_llm(client: Groq, anchor: str, A: str, B: str, model: str) -> Dict:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_prompt(anchor, A, B)}
    ]
    # Try Structured Outputs first
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=msgs,
            response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
            max_tokens=200,
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
    except Exception:
        # Fallback to JSON mode
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=msgs,
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)

    choice = data.get("choice", "").strip()
    if choice not in ("A","B"):
        choice = "A"
    aspects = data.get("aspects") or {}
    aspects = {
        "abstract_theme": bool(aspects.get("abstract_theme", False)),
        "course_of_action": bool(aspects.get("course_of_action", False)),
        "outcomes": bool(aspects.get("outcomes", False))
    }
    return {"choice": choice, "scores": {}, "aspects": aspects}

# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser(description="Task A adapted to Groq (embeddings or LLM judge).")
    ap.add_argument("--infile", default="data/dev_track_a.jsonl", help="Input JSONL with fields: id, anchor_text, text_a, text_b[, text_a_is_closer]")
    ap.add_argument("--outfile", default="output/track_a_groq.jsonl", help="Rich predictions JSONL")
    ap.add_argument("--submission", default="output/track_a.jsonl", help="Minimal submission JSONL")
    ap.add_argument("--method", choices=["embeddings","llm"], default=os.environ.get("TASKA_METHOD","llm"))
    ap.add_argument("--emb-model", default=os.environ.get("GROQ_EMB_MODEL","nomic-embed-text-v1.5"))
    ap.add_argument("--model", default=os.environ.get("GROQ_model","llama-3.1-8b-instant"))
    args = ap.parse_args()

    client = get_client()

    preds = []
    submit = []
    gold_bool = []
    pred_bool = []
    count = 1

    for ex in read_jsonl(args.infile):
        #print(ex)
        # Input field names expected by baseline repo
        anchor = ex.get("anchor_text") or ex.get("anchor") or ""
        A = ex.get("text_a") or ex.get("A") or ""
        B = ex.get("text_b") or ex.get("B") or ""
        print(f"Processing ID {count}...")

        if args.method == "embeddings":
            out = predict_with_embeddings(client, anchor, A, B, args.emb_model)
        else:
            out = predict_with_llm(client, anchor, A, B, args.model)

        preds.append({
            "id": count,
            "pred": out["choice"],
            "aspects_pred": out.get("aspects"),
            "scores": out.get("scores")
        })
        submit.append({"id": count, "label": out["choice"]})
        count += 1
        # For quick accuracy if dev has boolean gold
        if "text_a_is_closer" in ex and isinstance(ex["text_a_is_closer"], bool):
            gold_bool.append(ex["text_a_is_closer"])
            pred_bool.append(True if out["choice"] == "A" else False)

    # Write outputs
    write_jsonl(args.outfile, preds)
    write_jsonl(args.submission, submit)

    if gold_bool:
        gold = np.asarray(gold_bool, dtype=int)
        pred = np.asarray(pred_bool, dtype=int)
        acc = float((gold == pred).mean())
        print(f"Accuracy (vs text_a_is_closer): {acc:.3f}")
    print(f"Wrote {len(preds)} rows to {args.outfile} and {args.submission}")

if __name__ == "__main__":
    main()
