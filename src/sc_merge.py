#!/usr/bin/env python3
"""
SC-merge 自一致性合并: 把 n 个采样 run 的卡片按 issue_name 语义聚类,
频次 >= F 的类保留, 代表卡取众数 (名字/立场/证据链/未来论点), 按频次降序 cap 张。
--single-best: 只选与其他 run 平均卡片名相似度最高的那个 run (消融对照)。
"""
import os, sys, json, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_scorer import BGE

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def mode(vals):
    if not vals: return None
    c = collections.Counter(vals)
    m = max(c.values())
    for v in vals:              # 按出现顺序 tie-break
        if c[v] == m: return v
    return None

def chain_key(card):
    ch = card.get("argument_chain") or []
    if isinstance(ch, str): ch = [ch]
    return tuple(str(c).strip() for c in ch if str(c).strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)   # 读 {prefix}.run0..run{n-1}
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--freq", type=int, default=2)
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--bge", default="/data/neg_opinion/models/bge-small-zh-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--single-best", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    runs = []
    for k in range(a.n):
        p = f"{a.prefix}.run{k}"
        runs.append({r["sample_id"]: r for r in load(p)} if os.path.exists(p) else {})
    sids = [r["sample_id"] for r in runs[0].values()]
    print(f"[merge] {len(sids)} samples, n={a.n}, freq={a.freq}, thr={a.thr}, cap={a.cap}, single_best={a.single_best}", flush=True)
    bge = BGE(a.bge, device=a.device)

    out = []
    stats = collections.Counter(); cards_per = collections.Counter(); empty_fb = 0
    for sid in sids:
        recs = [r.get(sid) for r in runs]
        if a.single_best:
            best, bs = None, -1.0
            for k, r in enumerate(recs):
                if not r: continue
                names = [(it.get("issue_name") or "").strip() for it in (r.get("issue_list") or [])]
                names = [nm for nm in names if nm]
                if not names: continue
                es = [bge.emb(nm) for nm in names]
                s = cnt = 0
                for k2, r2 in enumerate(recs):
                    if k2 == k or not r2: continue
                    for it2 in (r2.get("issue_list") or []):
                        nm2 = (it2.get("issue_name") or "").strip()
                        if not nm2: continue
                        e2 = bge.emb(nm2)
                        for e in es:
                            s += float(np.dot(e, e2)); cnt += 1
                if cnt and s / cnt > bs:
                    bs, best = s / cnt, r
            r = best or next((x for x in recs if x), None)
            if r is None: continue
            out.append({"sample_id": sid, "issue_list": r.get("issue_list") or [],
                        "future_argument": r.get("future_argument") or []})
            cards_per[len(r.get("issue_list") or [])] += 1
            continue

        clusters = []   # {names, embs, cards:[(run, idx, card)], first}
        for k, r in enumerate(recs):
            if not r: continue
            items = r.get("issue_list") or []
            futs = r.get("future_argument") or []
            for i, it in enumerate(items):
                nm = (it.get("issue_name") or "").strip()
                if not nm: continue
                e = bge.emb(nm)
                placed = False
                for cl in clusters:
                    if any(float(np.dot(e, e2)) >= a.thr for e2 in cl["embs"]):
                        cl["names"].append(nm); cl["embs"].append(e)
                        cl["cards"].append((k, i, it, (futs[i] if i < len(futs) else "")))
                        placed = True; break
                if not placed:
                    clusters.append({"names": [nm], "embs": [e],
                                     "cards": [(k, i, it, (futs[i] if i < len(futs) else ""))]})
        reps, fut_reps, freqs = [], [], []
        for cl in clusters:
            freq = len(set(k for k, _, _, _ in cl["cards"]))
            freqs.append(freq); stats[f"freq{freq}"] += 1
            if freq < a.freq: continue
            reps.append({
                "issue_name": mode([c.get("issue_name", "").strip() for _, _, c, _ in cl["cards"]]),
                "stance": mode([(c.get("stance") or "").strip().lower() for _, _, c, _ in cl["cards"]]),
                "argument_chain": list(mode([chain_key(c) for _, _, c, _ in cl["cards"]]) or []),
            })
            fut_reps.append(mode([f for _, _, _, f in cl["cards"]]) or "")
        if not reps:                     # 兜底: 全类频次不足 -> 按频次取top
            empty_fb += 1
            order = sorted(range(len(clusters)), key=lambda x: (-freqs[x], x))[:a.cap]
            freqs2 = []
            for x in order:
                cl = clusters[x]
                reps.append({
                    "issue_name": mode([c.get("issue_name", "").strip() for _, _, c, _ in cl["cards"]]),
                    "stance": mode([(c.get("stance") or "").strip().lower() for _, _, c, _ in cl["cards"]]),
                    "argument_chain": list(mode([chain_key(c) for _, _, c, _ in cl["cards"]]) or []),
                })
                fut_reps.append(mode([f for _, _, _, f in cl["cards"]]) or "")
                freqs2.append(freqs[x])
        else:
            freqs2 = [freqs[x] for x in range(len(clusters)) if freqs[x] >= a.freq]
        order = sorted(range(len(reps)), key=lambda x: (-freqs2[x], x))[:a.cap]
        issue_list = [reps[x] for x in order]
        future_argument = [fut_reps[x] for x in order]
        cards_per[len(issue_list)] += 1
        out.append({"sample_id": sid, "issue_list": issue_list, "future_argument": future_argument})

    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[merge] cards/sample dist: {dict(sorted(cards_per.items()))}", flush=True)
    print(f"[merge] cluster freq hist: {dict(sorted(stats.items()))}; empty_fallback={empty_fb}", flush=True)
    print(f"[merge] saved {a.out} ({len(out)} samples)", flush=True)

if __name__ == "__main__":
    main()
