import json, argparse, random, os
from pathlib import Path

def decide_flag(ex):
    # Prefer boolean if present, else derive from "label"
    if isinstance(ex.get("text_a_is_closer"), bool):
        return ex["text_a_is_closer"]
    if ex.get("label") in ("A", "B"):
        return ex["label"] == "A"
    return None  # no gold → skip

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", help="path to your synthetic triples jsonl")
    ap.add_argument("outfile", help="path to write the 200+200 balanced jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=200, help="per-class count (default 200)")
    args = ap.parse_args()

    random.seed(args.seed)
    A, B = [], []
    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            ex = json.loads(line)
            flag = decide_flag(ex)
            if flag is None:
                continue
            (A if flag else B).append(ex)

    need = args.k
    if len(A) < need or len(B) < need:
        raise SystemExit(f"Not enough examples: A={len(A)}, B={len(B)}, need {need} each")

    A_sel = random.sample(A, need)
    B_sel = random.sample(B, need)
    out = A_sel + B_sel
    random.shuffle(out)

    Path(os.path.dirname(args.outfile) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.outfile, "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"wrote {len(out)} rows → {args.outfile} (A={need}, B={need})")

if __name__ == "__main__":
    main()
