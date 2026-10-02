#!/usr/bin/env python3
"""零GPU分析: 议题名是否为原文短语 (子串 / 与-拼接 / 最长公共子串覆盖)"""
import os, sys, json, re, collections
import numpy as np

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def norm_ws(s): return re.sub(r"\s+", "", s or "")

def lcs_len(a, b):
    # 最长公共子串(连续) 长度, DP 滚动
    if not a or not b: return 0
    prev = [0] * (len(b) + 1); best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1); ai = a[i-1]
        for j in range(1, len(b) + 1):
            if ai == b[j-1]:
                cur[j] = prev[j-1] + 1
                if cur[j] > best: best = cur[j]
        prev = cur
    return best

def analyze(recs, docs, tag):
    stats = collections.Counter(); n = 0
    lcs_cov = []; parts_all = []; parts_hit = []
    len_name = []
    for r in recs:
        docn = norm_ws(docs.get(r["sample_id"], ""))
        for it in (r.get("issue_list") or []):
            name = norm_ws(it.get("issue_name") or "")
            if not name: continue
            n += 1; len_name.append(len(name))
            stats["total"] += 1
            if name in docn:
                stats["exact_substring"] += 1
            elif "与" in name:
                parts = [p for p in name.split("与") if p]
                if len(parts) >= 2:
                    hits = sum(1 for p in parts if p in docn)
                    parts_all.append(len(parts)); parts_hit.append(hits)
                    if hits == len(parts): stats["yu_all_parts_sub"] += 1
                    elif hits > 0: stats["yu_some_parts_sub"] += 1
                    else: stats["yu_no_parts_sub"] += 1
                else:
                    stats["other"] += 1
            else:
                stats["other"] += 1
            lcs_cov.append(lcs_len(name, docn) / len(name))
    out = {"tag": tag, "n": n,
           "exact_substring_rate": round(stats["exact_substring"] / n, 4),
           "yu_all_parts_sub": stats["yu_all_parts_sub"],
           "yu_some_parts_sub": stats["yu_some_parts_sub"],
           "yu_no_parts_sub": stats["yu_no_parts_sub"],
           "other": stats["other"],
           "lcs_coverage_mean": round(float(np.mean(lcs_cov)), 4),
           "lcs_coverage_p50": round(float(np.median(lcs_cov)), 4),
           "lcs_coverage>=0.6_rate": round(float(np.mean([c >= 0.6 for c in lcs_cov])), 4),
           "name_chars_p50": int(np.median(len_name))}
    if parts_all:
        out["yu_parts_hit_rate"] = round(sum(parts_hit) / sum(parts_all), 4)
    return out

def main():
    R = "/data/neg_opinion/"
    gold_v = load(R + "data/val.jsonl"); pred_v = load(R + "eval/pred_r32_val.jsonl")
    train = load(R + "data/train.jsonl")
    docs_v = {r["sample_id"]: ((r.get("docs") or [{}])[0].get("full_text") or "") for r in gold_v}
    docs_t = {r["sample_id"]: ((r.get("docs") or [{}])[0].get("full_text") or "") for r in train}
    for recs, docs, tag in ((gold_v, docs_v, "val_gold"), (pred_v, docs_v, "val_pred"),
                            (train, docs_t, "train_gold")):
        print(json.dumps(analyze(recs, docs, tag), ensure_ascii=False))

if __name__ == "__main__":
    main()
