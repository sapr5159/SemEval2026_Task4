# data_audit.py  — no external deps (pure stdlib)
import json, re, math, statistics, argparse, os
from pathlib import Path
from collections import Counter

def tokenize(text): return re.findall(r"[A-Za-z0-9']+", (text or "").lower())
def tf(tokens): from collections import Counter; return Counter(tokens)
def cos_tf(tf1, tf2):
    if not tf1 or not tf2: return 0.0
    keys = set(tf1) | set(tf2)
    dot = sum(tf1.get(k,0)*tf2.get(k,0) for k in keys)
    n1 = math.sqrt(sum(v*v for v in tf1.values())); n2 = math.sqrt(sum(v*v for v in tf2.values()))
    return 0.0 if (n1==0 or n2==0) else dot/(n1*n2)

def read_jsonl(p):
    with open(p,"r",encoding="utf-8") as f:
        for i,line in enumerate(f, start=1):
            line=line.strip()
            if not line: continue
            try:
                yield i, json.loads(line)
            except Exception:
                yield i, None

def audit_track_a(path: Path):
    stats = {"total":0,"bad_json":0,"missing_fields":0,"id_dupes":0,"invalid_labels":0,
             "label_counts":Counter(),"aspects_present":0,"aspects_complete":0,"aspects_invalid":0,
             "len_chars":[], "len_tokens":[], "field_usage":Counter()}
    ids=set(); dupes=0; issues=[]
    gold=[]; pred=[]; ties=0
    for ln,obj in read_jsonl(path):
        if obj is None: stats["bad_json"]+=1; issues.append(("bad_json",ln)); continue
        stats["total"]+=1
        anchor = obj.get("anchor") or obj.get("anchor_text")
        A = obj.get("A") or obj.get("text_a")
        B = obj.get("B") or obj.get("text_b")
        if anchor is None or A is None or B is None: stats["missing_fields"]+=1; issues.append(("missing_fields",ln))
        _id = obj.get("id")
        if _id is None: issues.append(("missing_id",ln))
        else:
            if _id in ids: dupes+=1
            ids.add(_id)
        lab = obj.get("label")
        if lab is None or lab not in ("A","B"): stats["invalid_labels"]+=1
        else: stats["label_counts"][lab]+=1
        asp = obj.get("aspects")
        if asp is not None:
            stats["aspects_present"]+=1
            keys={"abstract_theme","course_of_action","outcomes"}
            if all(k in asp for k in keys) and all(isinstance(asp[k],bool) for k in keys):
                stats["aspects_complete"]+=1
            else:
                stats["aspects_invalid"]+=1
        if isinstance(anchor,str):
            stats["len_chars"].append(len(anchor))
            stats["len_tokens"].append(len(tokenize(anchor)))
        for k in obj.keys(): stats["field_usage"][k]+=1
        # lexical baseline
        if isinstance(anchor,str) and isinstance(A,str) and isinstance(B,str) and lab in ("A","B"):
            sA = cos_tf(tf(tokenize(anchor)), tf(tokenize(A)))
            sB = cos_tf(tf(tokenize(anchor)), tf(tokenize(B)))
            if abs(sA-sB)<1e-12: ties+=1; pred.append("A")
            else: pred.append("A" if sA>sB else "B")
            gold.append(lab)
    stats["id_dupes"]=dupes
    base=None
    if gold and len(gold)==len(pred):
        corr=sum(1 for g,p in zip(gold,pred) if g==p)
        base={"count":len(gold),"accuracy":round(corr/len(gold),4),"ties":ties}
    return stats, base, issues

def audit_track_b(path: Path):
    stats = {"total":0,"bad_json":0,"missing_fields":0,"id_dupes":0,"has_dev_label":0,
             "len_chars_anchor":[], "len_tokens_anchor":[], "field_usage":Counter()}
    ids=set(); dupes=0; issues=[]
    gold=[]; pred=[]; ties=0
    for ln,obj in read_jsonl(path):
        if obj is None: stats["bad_json"]+=1; issues.append(("bad_json",ln)); continue
        stats["total"]+=1
        anchor = obj.get("anchor_text") or obj.get("anchor")
        A = obj.get("text_a") or obj.get("A")
        B = obj.get("text_b") or obj.get("B")
        _id = obj.get("id")
        if _id is None: issues.append(("missing_id",ln))
        else:
            if _id in ids: dupes+=1
            ids.add(_id)
        if anchor is None or A is None or B is None: stats["missing_fields"]+=1; issues.append(("missing_fields",ln))
        if isinstance(anchor,str):
            stats["len_chars_anchor"].append(len(anchor))
            stats["len_tokens_anchor"].append(len(tokenize(anchor)))
        for k in (obj.keys() if isinstance(obj,dict) else []): stats["field_usage"][k]+=1
        if isinstance(obj.get("text_a_is_closer"),bool):
            stats["has_dev_label"]+=1
            sA = cos_tf(tf(tokenize(anchor or "")), tf(tokenize(A or "")))
            sB = cos_tf(tf(tokenize(anchor or "")), tf(tokenize(B or "")))
            if abs(sA-sB)<1e-12: ties+=1; pred.append(True)
            else: pred.append(sA>sB)
            gold.append(bool(obj["text_a_is_closer"]))
    stats["id_dupes"]=dupes
    base=None
    if gold:
        corr=sum(1 for g,p in zip(gold,pred) if g==p)
        base={"count":len(gold),"accuracy":round(corr/len(gold),4),"ties":ties}
    return stats, base, issues

def summarize_A(stats, base):
    out=[]
    out.append(f"- Total: {stats['total']}, Bad JSON: {stats['bad_json']}, Missing anchor/A/B: {stats['missing_fields']}, Duplicate IDs: {stats['id_dupes']}")
    out.append(f"- Label counts: {dict(stats['label_counts'])} | Invalid label rows: {stats['invalid_labels']}")
    out.append(f"- Aspects present: {stats['aspects_present']} (complete: {stats['aspects_complete']}, invalid: {stats['aspects_invalid']})")
    if stats["len_chars"]:
        out.append(f"- Anchor length (chars): mean={round(statistics.mean(stats['len_chars']),2)}, min={min(stats['len_chars'])}, max={max(stats['len_chars'])}")
    if stats["len_tokens"]:
        out.append(f"- Anchor length (tokens): mean={round(statistics.mean(stats['len_tokens']),2)}, min={min(stats['len_tokens'])}, max={max(stats['len_tokens'])}")
    if base: out.append(f"- Lexical baseline (TF-cosine) on gold N={base['count']}: acc={base['accuracy']}, ties={base['ties']}")
    out.append(f"- Top fields: {stats['field_usage'].most_common(10)}")
    return "\n".join(out)

def summarize_B(stats, base):
    out=[]
    out.append(f"- Total: {stats['total']}, Bad JSON: {stats['bad_json']}, Missing anchor/text_a/text_b: {stats['missing_fields']}, Duplicate IDs: {stats['id_dupes']}")
    out.append(f"- Rows with dev label (text_a_is_closer): {stats['has_dev_label']}")
    if stats["len_chars_anchor"]:
        out.append(f"- Anchor length (chars): mean={round(statistics.mean(stats['len_chars_anchor']),2)}, min={min(stats['len_chars_anchor'])}, max={max(stats['len_chars_anchor'])}")
    if stats["len_tokens_anchor"]:
        out.append(f"- Anchor length (tokens): mean={round(statistics.mean(stats['len_tokens_anchor']),2)}, min={min(stats['len_tokens_anchor'])}, max={max(stats['len_tokens_anchor'])}")
    if base: out.append(f"- Lexical baseline (TF-cosine) on gold N={base['count']}: acc={base['accuracy']}, ties={base['ties']}")
    out.append(f"- Top fields: {stats['field_usage'].most_common(10)}")
    return "\n".join(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--track_a", default="data/dev_track_a.jsonl")
    ap.add_argument("--track_b", default="data/dev_track_b.jsonl")
    args=ap.parse_args()

    rep = ["# Data Audit Report\n"]
    if Path(args.track_a).exists():
        a_stats, a_base, a_issues = audit_track_a(Path(args.track_a))
        rep += ["## Track A", summarize_A(a_stats,a_base)]
        if a_issues[:10]: rep.append(f"- First 10 issues: {a_issues[:10]}")
    else:
        rep += ["## Track A", "- File not found."]

    if Path(args.track_b).exists():
        b_stats, b_base, b_issues = audit_track_b(Path(args.track_b))
        rep += ["\n## Track B", summarize_B(b_stats,b_base)]
        if b_issues[:10]: rep.append(f"- First 10 issues: {b_issues[:10]}")
    else:
        rep += ["\n## Track B", "- File not found."]

    print("\n".join(rep))

if __name__ == "__main__":
    main()
