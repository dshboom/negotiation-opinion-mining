#!/usr/bin/env python3
"""⑥a 无标签先验匹配: 从 rescore 立场概率拟合全局先验偏置 (与 postprocess.py 同配方)。
用法: fit_bias.py --rescore eval/rescore_X.jsonl --out outputs/diag/X.bias.json"""
import json, argparse
import numpy as np

GLOBAL_PRIOR = {"support": 0.632, "neutral": 0.342, "oppose": 0.026}

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plist = []
    for r in load(a.rescore):
        for c in (r.get("cards") or []):
            plist.append([c["p_support"], c["p_neutral"], c["p_oppose"]])
    P = np.array(plist)
    tgt = np.array([GLOBAL_PRIOR[k] for k in ("support", "neutral", "oppose")])
    best, bd = (0.0, 0.0), 1e9
    for bn in np.arange(-1.0, 1.01, 0.1):
        for bo in np.arange(-0.5, 3.01, 0.1):
            lab = (P * np.exp([0.0, bn, bo])).argmax(1)
            dist = np.array([(lab == k).mean() for k in range(3)])
            d = float(np.abs(dist - tgt).sum())
            if d < bd:
                bd, best = d, (float(bn), float(bo))
    out = {"b_neutral": round(best[0], 3), "b_oppose": round(best[1], 3),
           "L1": round(bd, 4), "n": len(plist)}
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    print("[fit_bias]", out, flush=True)

if __name__ == "__main__":
    main()
