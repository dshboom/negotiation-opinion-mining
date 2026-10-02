#!/usr/bin/env python3
"""把 ⑥ 立场校准偏置应用到预测文件。"""
import argparse, json, os
import numpy as np
ST = ["support", "neutral", "oppose"]

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--rescore", required=True)
    ap.add_argument("--b-neutral", type=float, default=0.0)
    ap.add_argument("--b-oppose", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-array", default=None)
    a = ap.parse_args()
    pred = load(a.pred); rs = {r["sample_id"]: r for r in load(a.rescore)}
    b = np.array([0.0, a.b_neutral, a.b_oppose])
    changed = 0; total = 0
    out = []
    for r in pred:
        cards = (rs.get(r["sample_id"]) or {}).get("cards", [])
        items = []
        for i, it in enumerate(r.get("issue_list") or []):
            ni = dict(it)
            if i < len(cards):
                c = cards[i]
                sc = np.array([c["p_support"], c["p_neutral"], c["p_oppose"]]) * np.exp(b)
                ns = ST[int(sc.argmax())]
                total += 1
                if ns != ni.get("stance"): changed += 1
                ni["stance"] = ns
            items.append(ni)
        out.append({"sample_id": r["sample_id"], "issue_list": items,
                    "future_argument": list(r.get("future_argument") or [])})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.out_array:
        json.dump(out, open(a.out_array, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"applied biases (neutral={a.b_neutral}, oppose={a.b_oppose}); changed {changed}/{total} stances -> {a.out}")

if __name__ == "__main__":
    main()
