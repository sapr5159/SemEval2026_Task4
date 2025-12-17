# make_balanced_subset.py
import json, argparse, random, os
from pathlib import Path

def decide_flag(ex):
    # Prefer boolean if present, else derive from "label"
    if isinstance(ex.get("text_a_is_closer"), bool):
        return ex["text_a_is_closer"]
    if ex.get("label") in ("A", "B"):
        return ex["label"] == "A"
    return None  # no gold → skip

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser(
        description="Merge dev data with K synthetic rows and auto-name the output."
    )
    ap.add_argument("--infile", default=".\\data\\synthetic_data_for_classification.jsonl", help="path to synthetic triples jsonl")
    ap.add_argument("--devData", default=".\\data\\dev_track_a.jsonl", help="path to dev_track_a.jsonl (200 rows)")
    ap.add_argument("--outfile", default=None,
                    help="optional explicit output path; if omitted, saves as <dev_stem>_plus_<k>.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=100, help="TOTAL number of synthetic rows to add")
    args = ap.parse_args()

    random.seed(args.seed)

    # ---- load dev ----
    dev_path = Path(args.devData)
    dev_rows = load_jsonl(dev_path)
    n_dev = len(dev_rows)

    # ---- load synth and split by gold ----
    A, B = [], []
    for ex in load_jsonl(args.infile):
        flag = decide_flag(ex)
        if flag is None:
            continue
        (A if flag else B).append(ex)

    # ---- choose exactly k synthetic rows, approximately balanced ----
    k_total = max(0, int(args.k))
    if k_total == 0:
        synth_sel = []
    else:
        a_need = k_total // 2
        b_need = k_total - a_need

        a_pick = min(a_need, len(A))
        b_pick = min(b_need, len(B))

        # backfill if one class is short
        deficit = k_total - (a_pick + b_pick)
        if deficit > 0:
            # choose from whichever side has more remaining
            pool = (A if (len(A) - a_pick) >= (len(B) - b_pick) else B)
            extra_pick = min(deficit, len(pool) - (a_pick if pool is A else b_pick))
            if pool is A:
                a_pick += max(0, extra_pick)
            else:
                b_pick += max(0, extra_pick)

        if a_pick + b_pick < k_total:
            raise SystemExit(
                f"Not enough synthetic examples to reach k={k_total}: "
                f"A_avail={len(A)}, B_avail={len(B)}, picked A={a_pick}, B={b_pick}"
            )

        synth_sel = random.sample(A, a_pick) + random.sample(B, b_pick)
        random.shuffle(synth_sel)

    # ---- decide output path ----
    if args.outfile:
        out_path = Path(args.outfile)
    else:
        # save next to devData, named as <dev_stem>_plus_<k>.json (per your request)
        out_name = f"{dev_path.stem}_plus_{k_total}.json"
        out_path = dev_path.parent / out_name

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- write dev + synth (JSONL content; extension is .json by request) ----
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in dev_rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        for ex in synth_sel:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(
        f"wrote {n_dev + len(synth_sel)} rows → {out_path} "
        f"(dev={n_dev}, synth={len(synth_sel)}; picked A≈{sum(decide_flag(x) is True for x in synth_sel)}, "
        f"B≈{sum(decide_flag(x) is False for x in synth_sel)})"
    )

if __name__ == "__main__":
    main()
