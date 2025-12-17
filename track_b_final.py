#!/usr/bin/env python3
import os, json, argparse, numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- utils ----------------
def norm(s:str)->str: return " ".join((s or "").split())
def read_jsonl(p:str):
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)

def cos(a:np.ndarray,b:np.ndarray)->float:
    # works with either pre-normalized or raw; re-normalize just in case
    a = a.astype("float32"); b = b.astype("float32")
    a = a/(np.linalg.norm(a)+1e-12); b = b/(np.linalg.norm(b)+1e-12)
    return float(a@b)

# ------------- ST backend (optional re-embed) -------------
_ST = None; _ST_NAME=None
def st_load(name:str):
    global _ST,_ST_NAME
    from sentence_transformers import SentenceTransformer
    if _ST is None or _ST_NAME!=name:
        _ST = SentenceTransformer(name, trust_remote_code=True)
        _ST_NAME=name
    return _ST

def st_embed(texts:List[str], model_name:str, batch:int, normalize=True)->np.ndarray:
    m = st_load(model_name)
    V = m.encode(texts, batch_size=batch, normalize_embeddings=normalize, show_progress_bar=False)
    return np.asarray(V, dtype="float32")

def prefixify(texts:List[str], role:str, mode:str)->List[str]:
    if mode!="bge": return texts
    if role=="anchor": return [f"query: {t}" for t in texts]
    return [f"passage: {t}" for t in texts]

def chunk_mean(texts:List[str], max_chars:int, overlap:int)->List[str]:
    # simple sentence-ish chunking and join; we’ll just treat each text as is if short
    import re
    out=[]
    for t in texts:
        if max_chars<=0 or len(t)<=max_chars:
            out.append(t); continue
        sents = re.split(r"(?<=[.!?])\s+", t.strip()); sents=[s for s in sents if s]
        cur=[]; cur_len=0; chunks=[]
        for s in sents:
            if cur_len+len(s)+1 <= max_chars:
                cur.append(s); cur_len += len(s)+1
            else:
                chunks.append(" ".join(cur)); cur=[s]; cur_len=len(s)
        if cur: chunks.append(" ".join(cur))
        # mean of embeddings -> we’ll keep text form here; actual mean happens after encoding
        out.append(" ||| ".join(chunks))  # marker; handled in encode step
    return out

def encode_with_chunks_role(texts:List[str], role:str, model:str, prefix:str, chunk_chars:int, overlap:int, batch:int)->np.ndarray:
    # expand each text into chunk list (split on " ||| "), encode each part, then mean-pool
    pre = prefixify(texts, role, prefix)
    if chunk_chars>0:
        pre = chunk_mean(pre, chunk_chars, overlap)
    # flatten
    split_lists = [t.split(" ||| ") for t in pre]
    flat = [s for lst in split_lists for s in lst]
    E = st_embed(flat, model, batch, normalize=True)
    # mean per original text
    out=[]; i=0
    for lst in split_lists:
        k=len(lst); vec = E[i:i+k].mean(axis=0)
        vec = vec/(np.linalg.norm(vec)+1e-12)
        out.append(vec); i+=k
    return np.stack(out, axis=0)

# ---------------- core eval ----------------
def build_text_index_from_trackB(devB_path:str)->List[str]:
    # row order = corpus order; that’s what you embedded
    return [norm(obj.get("text","")) for obj in read_jsonl(devB_path)]

def load_trackB_vectors(emb_path:str, ids_path:str)->Tuple[np.ndarray,List[str]]:
    V = np.load(emb_path)  # (N, D)
    ids = [json.loads(l)["id"] for l in open(ids_path,"r",encoding="utf-8")]
    assert V.shape[0]==len(ids), "embeddings and ids length mismatch"
    return V, ids

def text_to_index_map(textsB:List[str])->Dict[str,int]:
    m={}
    for i,t in enumerate(textsB):
        if t and t not in m: m[t]=i
    return m

def eval_using_precomputed(devA:str, devB:str, emb:str, ids:str)->None:
    textsB = build_text_index_from_trackB(devB)
    V, _ = load_trackB_vectors(emb, ids)
    # if your V are unit vectors (they are), cosine = dot
    V = V.astype("float32")
    T2I = text_to_index_map(textsB)

    total=0; correct=0; miss=0
    for ex in read_jsonl(devA):
        a = norm(ex.get("anchor_text") or ex.get("anchor") or "")
        A = norm(ex.get("text_a") or ex.get("A") or "")
        B = norm(ex.get("text_b") or ex.get("B") or "")
        gold_lab = ex.get("label")
        hint = ex.get("text_a_is_closer")

        try:
            ia, iA, iB = T2I[a], T2I[A], T2I[B]
        except KeyError:
            miss += 1
            continue

        sA = float(V[ia] @ V[iA]); sB = float(V[ia] @ V[iB])
        pred_is_A = (sA >= sB)

        if gold_lab in ("A","B"):
            total += 1; correct += int(pred_is_A == (gold_lab=="A"))
        elif isinstance(hint,bool):
            total += 1; correct += int(pred_is_A == hint)

    print(f"[precomputed] used={total}, missing_triplets={miss}")
    if total>0:
        print(f"[precomputed] accuracy={correct/total:.3f}")

def eval_reembed_all(devA:str, model:str, prefix:str, chunk_chars:int, overlap:int, batch:int)->None:
    # re-embed anchor/A/B with improved scorer, independent of track_b.npy
    rows=list(read_jsonl(devA))
    anchors=[norm(r.get("anchor_text") or r.get("anchor") or "") for r in rows]
    As=[norm(r.get("text_a") or r.get("A") or "") for r in rows]
    Bs=[norm(r.get("text_b") or r.get("B") or "") for r in rows]

    VA = encode_with_chunks_role(anchors, "anchor", model, prefix, chunk_chars, overlap, batch)
    V1 = encode_with_chunks_role(As,      "doc",    model, prefix, chunk_chars, overlap, batch)
    V2 = encode_with_chunks_role(Bs,      "doc",    model, prefix, chunk_chars, overlap, batch)

    total=0; correct=0
    for r,va,vA,vB in zip(rows,VA,V1,V2):
        gold_lab = r.get("label"); hint = r.get("text_a_is_closer")
        
        scoresA = np.diag(cosine_similarity(VA, V1))
        scoresB = np.diag(cosine_similarity(VA, V2))
        pred_is_A = (cos(va,vA) >= cos(va,vB))
        if gold_lab in ("A","B"):
            total+=1; correct+=int(pred_is_A==(gold_lab=="A"))
        elif isinstance(hint,bool):
            total+=1; correct+=int(pred_is_A==hint)
    print(f"[reembed] used={total}, accuracy={correct/total:.3f}" if total>0 else "[reembed] no gold in file")

def main():
    ap = argparse.ArgumentParser(description="Evaluate Track-B cosine decisions on Track A using Track B embeddings (or improved re-embed).")
    ap.add_argument("--devA", default="data/dev_track_a.jsonl")
    ap.add_argument("--devB", default="data/dev_track_b.jsonl")
    ap.add_argument("--emb",  default="outputs/track_b.npy")
    ap.add_argument("--ids",  default="outputs/track_b_ids.jsonl")
    ap.add_argument("--reembed-all", action="store_true", help="Ignore precomputed embeddings; re-embed anchor/A/B with improved settings")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--prefix", choices=["none","bge"], default="none")
    ap.add_argument("--chunk-chars", type=int, default=0)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    if args.reembed_all:
        eval_reembed_all(args.devA, args.model, args.prefix, args.chunk_chars, args.chunk_overlap, args.batch)
    else:
        eval_using_precomputed(args.devA, args.devB, args.emb, args.ids)

if __name__ == "__main__":
    main()
