import json, sys
from collections import Counter

if len(sys.argv) < 2:
    print("Usage: python count_synth_balance.py <path/to/synthetic.jsonl>")
    sys.exit(1)

path = sys.argv[1]
c = Counter()
total = 0
skipped = 0

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: 
            continue
        try:
            ex = json.loads(line)
        except Exception:
            skipped += 1
            continue

        if isinstance(ex.get("text_a_is_closer"), bool):
            c["A" if ex["text_a_is_closer"] else "B"] += 1
            total += 1
        elif ex.get("label") in ("A", "B"):
            c[ex["label"]] += 1
            total += 1
        else:
            skipped += 1

print(f"total_with_gold : {total}")
print(f"text_a_closer   : {c['A']}")
print(f"text_b_closer   : {c['B']}")
if total:
    print(f"A%={c['A']/total:.3f}  B%={c['B']/total:.3f}")
if skipped:
    print(f"skipped (no label/flag): {skipped}")
