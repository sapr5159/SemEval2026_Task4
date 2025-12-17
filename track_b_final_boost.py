#!/usr/bin/env python3
# pip install sentence-transformers numpy
import os, json, argparse, re, numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# ---------- basic utils ----------
def read_jsonl(p:str):
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)

def norm(s:str)->str: return " ".join((s or "").split())
def cos(a,b):
    a=a.astype("float32"); b=b.astype("float32")
    a/= (np.linalg.norm(a)+1e-12); b/= (np.linalg.norm(b)+1e-12)
    return float(a@b)

# ---------- ABTT (All-but-the-top) ----------
def fit_abtt(V: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    V: (N,D) unit or not. Returns (mean, top_k_components) as (D,) and (k,D).
    """
    mu = V.mean(axis=0, keepdims=True)  # (1,D)
    X = V - mu
    # economy SVD on covariance via X (handles big D)
    U, S, VT = np.linalg.svd(X, full_matrices=False)  # X ~ U S VT
    comps = VT[:k] if k>0 else np.zeros((0, V.shape[1]), dtype=V.dtype)  # (k,D)
    return mu[0], comps

def transform_abtt(V: np.ndarray, mu: np.ndarray, comps: np.ndarray) -> np.ndarray:
    X = V - mu
    # remove projection onto each top component
    for u in comps:
        proj = (X @ u[:,None]) * u[None,:]  # (N,1)*(1,D) -> (N,D)
        X = X - proj
    # L2 normalize rows
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X.astype("float32")

# ---------- SentenceTransformers backend ----------
_ST = None; _ST_NAME=None
def st_load(name:str):
    global _ST,_ST_NAME
    from sentence_transformers import SentenceTransformer
    if _ST is None or _ST_NAME!=name:
        _ST = SentenceTransformer(name, trust_remote_code=True)
        _ST_NAME=name
    return _ST

def st_embed(texts:List[str], model:str, batch:int, normalize=True)->np.ndarray:
    m = st_load(model)
    V = m.encode(texts, batch_size=batch, normalize_embeddings=normalize, show_progress_bar=False)
    return np.asarray(V, dtype="float32")

# ---------- aspect views (cheap & rule-based) ----------
_VERB_RE = re.compile(r"\b\w+(ed|ing)\b", re.I)
def verbs_only(t:str)->str:
    toks = re.findall(r"[A-Za-z']+", t)
    verbs = [w for w in toks if _VERB_RE.search(w)]
    return " ".join(verbs) if verbs else t

def ending_sentences(t:str, n:int=2)->str:
    sents = re.split(r"(?<=[.!?])\s+", t.strip())
    sents = [s for s in sents if s]
    if not sents: return t
    return " ".join(sents[-n:])

def prefixify(texts:List[str], role:str, mode:str)->List[str]:
    if mode!="bge": return texts
    return [(("query: " if role=="anchor" else "passage: ") + x) for x in texts]

def aspect_fused_embed(texts:List[str], role:str, model:str, batch:int, prefix:str,
                       w_raw=0.5, w_verb=0.25, w_end=0.25)->np.ndarray:
    raw = [norm(x) for x in texts]
    v_only  = [verbs_only(x) for x in raw]
    ending  = [ending_sentences(x, n=2) for x in raw]

    raw = prefixify(raw, role, prefix)
    v_only = prefixify(v_only, role, prefix)
    ending = prefixify(ending, role, prefix)

    E_raw = st_embed(raw,   model, batch, normalize=True)
    E_v   = st_embed(v_only,model, batch, normalize=True)
    E_end = st_embed(ending,model, batch, normalize=True)

    # weighted sum -> renorm
    V = w_raw*E_raw + w_verb*E_v + w_end*E_end
    V = V / (np.linalg.norm(V,axis=1,keepdims=True)+1e-12)
    return V.astype("float32")

# ---------- evaluation paths ----------
def eval_reembed(devA:str, model:str, prefix:str, batch:int, abtt_k:int,
                 w_raw=0.5, w_verb=0.25, w_end=0.25):
    rows = list(read_jsonl(devA))
    anchors = [norm(r.get("anchor_text") or r.get("anchor") or "") for r in rows]
    As      = [norm(r.get("text_a") or r.get("A") or "") for r in rows]
    Bs      = [norm(r.get("text_b") or r.get("B") or "") for r in rows]

    VA = aspect_fused_embed(anchors, "anchor", model, batch, prefix, w_raw, w_verb, w_end)
    V1 = aspect_fused_embed(As,      "doc",    model, batch, prefix, w_raw, w_verb, w_end)
    V2 = aspect_fused_embed(Bs,      "doc",    model, batch, prefix, w_raw, w_verb, w_end)

    if abtt_k>0:
        # fit ABTT on all candidate/anchor vectors together (unsupervised)
        V_all = np.vstack([VA,V1,V2])
        mu, comps = fit_abtt(V_all, k=abtt_k)
        VA, V1, V2 = transform_abtt(VA,mu,comps), transform_abtt(V1,mu,comps), transform_abtt(V2,mu,comps)

    total=0; correct=0
    for r,va,vA,vB in zip(rows,VA,V1,V2):
        gold = r.get("label"); hint = r.get("text_a_is_closer")
        pred_is_A = (cos(va,vA) >= cos(va,vB))
        if gold in ("A","B"):
            total+=1; correct+=int(pred_is_A==(gold=="A"))
        elif isinstance(hint,bool):
            total+=1; correct+=int(pred_is_A==hint)
    print(f"[reembed+fusion+abtt] used={total}, accuracy={correct/total:.3f}" if total>0 else "no gold found")

def eval_precomputed(devA:str, devB:str, emb:str, ids:str, abtt_k:int):
    # reuse your saved Track-B vectors; optional ABTT only
    corpus = [norm(o.get("text","")) for o in read_jsonl(devB)]
    V = np.load(emb).astype("float32")
    ids_list = [json.loads(l)["id"] for l in open(ids,"r",encoding="utf-8")]
    assert V.shape[0]==len(ids_list)==len(corpus)

    idx = {t:i for i,t in enumerate(corpus)}
    if abtt_k>0:
        mu, comps = fit_abtt(V, k=abtt_k)
        V = transform_abtt(V, mu, comps)

    total=0; correct=0; miss=0
    for ex in read_jsonl(devA):
        a = norm(ex.get("anchor_text") or ex.get("anchor") or "")
        A = norm(ex.get("text_a") or ex.get("A") or "")
        B = norm(ex.get("text_b") or ex.get("B") or "")
        gold = ex.get("label"); hint = ex.get("text_a_is_closer")
        try:
            ia,iA,iB = idx[a], idx[A], idx[B]
        except KeyError:
            miss += 1
            continue
        pred_is_A = (cos(V[ia],V[iA]) >= cos(V[ia],V[iB]))
        if gold in ("A","B"):
            total+=1; correct+=int(pred_is_A==(gold=="A"))
        elif isinstance(hint,bool):
            total+=1; correct+=int(pred_is_A==hint)
    print(f"[precomputed+abtt] used={total}, missing_triplets={miss}, accuracy={(correct/total if total else float('nan')):.3f}")

def main():
    ap = argparse.ArgumentParser(description="Boost Track-B cosine with ABTT + aspect fusion")
    ap.add_argument("--reembed-all", action="store_true", help="Ignore precomputed embeddings; re-embed anchor/A/B with improved settings")
    ap.add_argument("--devA", default="data/dev_track_a.jsonl")
    ap.add_argument("--devB", default="data/dev_track_b.jsonl")
    ap.add_argument("--emb",  default="outputs/track_b.npy")
    ap.add_argument("--ids",  default="outputs/track_b_ids.jsonl")
    ap.add_argument("--model", default="intfloat/e5-large-v2")
    ap.add_argument("--prefix", choices=["none","bge"], default="bge")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--abtt-k", type=int, default=1, help="remove top-k PCs (0 to disable)")
    ap.add_argument("--w-raw",  type=float, default=0.5)
    ap.add_argument("--w-verb", type=float, default=0.25)
    ap.add_argument("--w-end",  type=float, default=0.25)
    ap.add_argument("--add-synthetic", default=0, type=int, help="number of synthetic rows to add to devA for analysis")
    args = ap.parse_args()

    if args.add_synthetic>0:
        args.devA = f"data/dev_track_a_plus_{args.add_synthetic}.json"
        print(f"Note: devA overridden to {args.devA} to include +{args.add_synthetic} synthetic rows", args.devA)
    if args.reembed_all:
        eval_reembed(args.devA, args.model, args.prefix, args.batch, args.abtt_k,
                     args.w_raw, args.w_verb, args.w_end)
    else:
        eval_precomputed(args.devA, args.devB, args.emb, args.ids, args.abtt_k)

if __name__=="__main__":
    main()
