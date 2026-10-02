#!/usr/bin/env python3
"""
P1: 本地评分器 —— 复刻官方评分逻辑 (README §3 / §5.3)

综合分 = 0.8 * 观点抽取 + 0.2 * 未来观点预测

抽取:
  1. 立场必须严格相等, 否则该对不可匹配 (硬门槛)
  2. 三元组语义等价分 = 0.4*cos(议题名,bge) + 0.6*cos(证据链,bge)
  3. 最大二部图匹配 (scipy.optimize.linear_sum_assignment)
  4. NC = 匹配对中 语义等价分 > 0.7 的数量
  5. P=NC/Np, R=NC/Ng, F1=2PR/(P+R)
  6. alpha = 成功匹配对的 BERTScore-F1 均值; 抽取分 *= alpha

预测:
  - 复用抽取匹配; 每条预测按 ROUGE-L / BERTScore 与对应 gold 比较
  - 分母 = max(Ng, Np)

用法:
  python eval_scorer.py --gold val.jsonl --pred preds.jsonl [--out report.json]
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.optimize import linear_sum_assignment

# ----------------------------- IO -----------------------------
def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def norm_triples(items):
    res = []
    for it in items or []:
        chain = it.get("argument_chain") or []
        if isinstance(chain, str):
            chain = [chain]
        res.append({
            "issue_name": (it.get("issue_name") or "").strip(),
            "stance": (it.get("stance") or "").strip().lower(),
            "chain": [str(c).strip() for c in chain if str(c).strip()],
        })
    return res

def chain_text(t):
    return " ".join(t["chain"])

def triple_text(t):
    return (t["issue_name"] + " " + chain_text(t)).strip()

# ----------------------------- bge -----------------------------
class BGE:
    def __init__(self, path, device=None):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(path, device=device)
        self._cache = {}
    def emb(self, text):
        v = self._cache.get(text)
        if v is None:
            v = self.m.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
            self._cache[text] = v
        return v
    def cos(self, a, b):
        if not a or not b:
            return 0.0
        return float(np.dot(self.emb(a), self.emb(b)))

# ----------------------------- ROUGE-L -----------------------------
def rouge_l_f1(cand, ref):
    if not cand or not ref:
        return 0.0
    a, b = list(cand), list(ref)
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[len(b)]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r) if (p + r) else 0.0

# ----------------------------- BERTScore (batched) -----------------------------
class BertScorer:
    def __init__(self, model_path, device="cpu", num_layers=9):
        self.model_path, self.device, self.num_layers = model_path, device, num_layers
    def f1_batch(self, cands, refs):
        """returns list[float] same length; empty strings -> 0"""
        if not cands:
            return []
        import bert_score
        idx = [i for i, (c, r) in enumerate(zip(cands, refs)) if c and r]
        out = [0.0] * len(cands)
        if idx:
            _, _, F = bert_score.score([cands[i] for i in idx], [refs[i] for i in idx],
                                       model_type=self.model_path, num_layers=self.num_layers,
                                       verbose=False, device=self.device, lang="zh",
                                       rescale_with_baseline=False)
            for k, i in enumerate(idx):
                out[i] = float(F[k])
        return out

# ----------------------------- 核心 -----------------------------
def triple_equiv(p, g, bge):
    si = bge.cos(p["issue_name"], g["issue_name"])
    se = bge.cos(chain_text(p), chain_text(g))
    return 0.4 * si + 0.6 * se

def match_and_extract(preds, golds, bge, threshold=0.7, match_mode="sum"):
    Np, Ng = len(preds), len(golds)
    if Np == 0 or Ng == 0:
        return 0, Np, Ng, [], np.zeros((Np, Ng))
    sim = np.zeros((Np, Ng))
    ok = np.zeros((Np, Ng), dtype=bool)
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            ok[i, j] = (p["stance"] == g["stance"])
            if ok[i, j]:
                sim[i, j] = triple_equiv(p, g, bge)
    if match_mode == "threshold":
        ri, ci = linear_sum_assignment(-((sim > threshold) & ok).astype(float))
    else:
        mat = sim.copy(); mat[~ok] = -1e9
        ri, ci = linear_sum_assignment(-mat)
    pairs = [(int(i), int(j), float(sim[i, j])) for i, j in zip(ri, ci) if ok[i, j]]
    NC = sum(1 for _, _, s in pairs if s > threshold)
    return NC, Np, Ng, pairs, sim

def evaluate(gold_items, pred_items, bge, bs, threshold=0.7, match_mode="sum",
             alpha_mode="mul", pred_score_mode="mean", limit=None):
    gmap = {g["sample_id"]: g for g in gold_items}
    pmap = {p["sample_id"]: p for p in pred_items}
    ids = [g["sample_id"] for g in gold_items]
    if limit:
        ids = ids[:limit]

    NC_t = Np_t = Ng_t = 0
    per = []
    # 收集所有匹配对, 便于批量 BERTScore
    alpha_pairs = []      # (cand, ref)
    fut_pairs = []        # (pc, gc)
    sample_meta = []      # (sid, NC, Np, Ng, denom, npairs)
    for sid in ids:
        g = gmap[sid]
        p = pmap.get(sid, {"issue_list": [], "future_argument": []})
        golds = norm_triples(g.get("issue_list"))
        preds = norm_triples(p.get("issue_list"))
        NC, Np, Ng, pairs, sim = match_and_extract(preds, golds, bge, threshold, match_mode)
        gf = g.get("future_argument") or []
        pf = p.get("future_argument") or []
        base = len(alpha_pairs)
        for i, j, s in pairs:
            alpha_pairs.append((triple_text(preds[i]), triple_text(golds[j])))
            fut_pairs.append((str(pf[i]) if i < len(pf) else "",
                              str(gf[j]) if j < len(gf) else ""))
        NC_t += NC; Np_t += Np; Ng_t += Ng
        sample_meta.append((sid, NC, Np, Ng, max(Ng, Np) or 1, len(pairs), base))
        if len(sample_meta) % 50 == 0:
            print(f"  match {len(sample_meta)}/{len(ids)}", flush=True)

    print(f"  [bertscore] batches: alpha={len(alpha_pairs)} fut={len(fut_pairs)}", flush=True)
    alpha_scores = bs.f1_batch([a for a, _ in alpha_pairs], [b for _, b in alpha_pairs])
    fut_scores = bs.f1_batch([a for a, _ in fut_pairs], [b for _, b in fut_pairs])

    rouge_vals, bert_vals, alpha_vals = [], [], []
    for (sid, NC, Np, Ng, denom, npairs, base) in sample_meta:
        # alpha: 该样本成功匹配对的 BERTScore 均值
        a_scores = alpha_scores[base:base + npairs]
        alpha = float(np.mean(a_scores)) if a_scores else 0.0
        alpha_vals.append(alpha)
        r_sum = b_sum = 0.0
        for k in range(npairs):
            pc, gc = fut_pairs[base + k]
            r_sum += rouge_l_f1(pc, gc)
            b_sum += fut_scores[base + k]
        rouge_vals.append(r_sum / denom)
        bert_vals.append(b_sum / denom)
        per.append({"sample_id": sid, "NC": NC, "Np": Np, "Ng": Ng, "alpha": alpha,
                    "rouge": r_sum / denom, "bscore": b_sum / denom})

    P = NC_t / Np_t if Np_t else 0.0
    R = NC_t / Ng_t if Ng_t else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    alpha = float(np.mean(alpha_vals)) if alpha_vals else 0.0
    extract = F1 * alpha if alpha_mode == "mul" else F1 + alpha
    rouge = float(np.mean(rouge_vals)) if rouge_vals else 0.0
    bsc = float(np.mean(bert_vals)) if bert_vals else 0.0
    pred = {"mean": 0.5 * (rouge + bsc), "rouge": rouge, "bert": bsc}[pred_score_mode]
    composite = 0.8 * extract + 0.2 * pred
    report = {"samples": len(sample_meta), "NC": NC_t, "Np": Np_t, "Ng": Ng_t,
              "precision": round(P, 4), "recall": round(R, 4), "F1": round(F1, 4),
              "alpha": round(alpha, 4), "extract_score": round(extract, 4),
              "rouge_l": round(rouge, 4), "bertscore": round(bsc, 4),
              "predict_score": round(pred, 4), "composite": round(composite, 4)}
    return report, per

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--bge", default="/data/neg_opinion/models/bge-small-zh-v1.5")
    ap.add_argument("--bert", default="/data/neg_opinion/models/bert-base-chinese")
    ap.add_argument("--bert-layers", type=int, default=9)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--match-mode", default="sum", choices=["sum", "threshold"])
    ap.add_argument("--alpha-mode", default="mul", choices=["mul", "add"])
    ap.add_argument("--pred-score-mode", default="mean", choices=["mean", "rouge", "bert"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-sample", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    gold = load_jsonl(args.gold); pred = load_jsonl(args.pred)
    print(f"[load] gold={len(gold)} pred={len(pred)}", flush=True)
    bge = BGE(args.bge, device=args.device)
    bs = BertScorer(args.bert, device=args.device, num_layers=args.bert_layers)
    report, per = evaluate(gold, pred, bge, bs, threshold=args.threshold,
                           match_mode=args.match_mode, alpha_mode=args.alpha_mode,
                           pred_score_mode=args.pred_score_mode, limit=args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[time] {time.time()-t0:.1f}s", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"report": report, "per_sample": per if args.per_sample else None},
                      f, ensure_ascii=False, indent=2)
        print("saved", args.out)

if __name__ == "__main__":
    main()
