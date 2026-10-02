#!/usr/bin/env python3
"""
Phase 0 诊断 (val, 当前操作点 = r32 + ⑥全局先验校准):
  D1 未匹配归因: 每张金标/预测卡没匹配上的原因 (立场无对 / 相似度不足 / 配对挤占 / 配对但低于阈值)
     + 近失带分布 + chain/issue 两个 oracle (诊断用上界, 有标签泄漏, 只做归因)
  D2 证据逐字率: pred/gold 的 argument_chain 是否为原文逐字子串
  D3 未来论点溯源: gold/pred 的 future_argument 是否来自原文 (定 ⑩ 走 a 选句 还是 b 重训)
  D4 ⑥变体扫尾: position 先验 / 置信门控, 全部无标签先验, FastMatcher 快评 F1
"""
import os, sys, json, re, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_scorer import BGE, norm_triples, chain_text
from scipy.optimize import linear_sum_assignment

STANCES = ["support", "neutral", "oppose"]
GLOBAL_BIAS = np.array([0.0, 0.7, -0.5])

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def load_map(p): return {r["sample_id"]: r for r in load(p)} if p and os.path.exists(p) else {}
def cards_of(rmap, sid): return (rmap.get(sid) or {}).get("cards", [])
def norm_ws(s): return re.sub(r"\s+", "", s or "")

def bigrams(s):
    s = norm_ws(s)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i+2] for i in range(len(s)-1)}

def copy_ratio(text, docbg):
    bg = bigrams(text)
    return len(bg & docbg) / len(bg) if bg else 0.0

def band(s):
    edges = [0.7, 0.65, 0.6, 0.5, 0.4, 0.0]
    for k in range(len(edges) - 1):
        if edges[k+1] < s <= edges[k]:
            return "(%.2f,%.2f]" % (edges[k+1], edges[k])
    return ">0.70"

def chain_items(it):
    ch = it.get("argument_chain") or []
    if isinstance(ch, str): ch = [ch]
    return [str(c).strip() for c in ch if str(c).strip()]

# ---------------- D1 ----------------
def d1(gold, pred, rmap, bge, stance_fn, tag):
    gmap = {g["sample_id"]: g for g in gold}
    pmap = {p["sample_id"]: p for p in pred}
    G_reason = collections.Counter(); P_reason = collections.Counter()
    G_stance_blocked = collections.Counter(); G_stance_total = collections.Counter()
    pred_stance_dist = collections.Counter()
    bands = collections.Counter(); oracle = collections.Counter()
    cos_issue_nm, cos_chain_nm = [], []
    examples = []
    NC = 0; Np_t = 0; Ng_t = 0
    for sid, g in gmap.items():
        p = pmap.get(sid, {"issue_list": []})
        pg = norm_triples(g.get("issue_list")); pp = norm_triples(p.get("issue_list"))
        Np, Ng = len(pp), len(pg); Np_t += Np; Ng_t += Ng
        if Np == 0 or Ng == 0:
            for j in range(Ng): G_reason["no_pred_cards"] += 1
            for i in range(Np): P_reason["no_gold_cards"] += 1
            continue
        pst = [stance_fn(sid, i, pp[i]["stance"]) for i in range(Np)]
        gst = [t["stance"] for t in pg]
        for s in pst: pred_stance_dist[s] += 1
        for s in gst: G_stance_total[s] += 1
        sim = np.zeros((Np, Ng)); ok = np.zeros((Np, Ng), dtype=bool)
        for i in range(Np):
            for j in range(Ng):
                ok[i, j] = (pst[i] == gst[j])
                if ok[i, j]:
                    sim[i, j] = (0.4 * bge.cos(pp[i]["issue_name"], pg[j]["issue_name"])
                                 + 0.6 * bge.cos(chain_text(pp[i]), chain_text(pg[j])))
        mat = sim.copy(); mat[~ok] = -1e9
        ri, ci = linear_sum_assignment(-mat)
        assign_g, assign_p = {}, {}
        for i, j in zip(ri, ci):
            if ok[i, j]:
                assign_g[int(j)] = int(i); assign_p[int(i)] = int(j)
        for j in range(Ng):
            i = assign_g.get(j)
            if i is not None and sim[i, j] > 0.7:
                G_reason["matched"] += 1; NC += 1; continue
            partners = [x for x in range(Np) if ok[x, j]]
            if not partners:
                G_reason["1_no_stance_partner"] += 1
                G_stance_blocked[gst[j]] += 1
                continue
            bi = max(partners, key=lambda x: sim[x, j]); s = sim[bi, j]
            if i is not None: G_reason["3_assigned_below_thr"] += 1
            elif s > 0.7: G_reason["4_pairing_collision"] += 1
            else: G_reason["2_sim_below"] += 1
            if s <= 0.7: bands[band(float(s))] += 1
            ci_ = bge.cos(pp[bi]["issue_name"], pg[j]["issue_name"])
            cc_ = bge.cos(chain_text(pp[bi]), chain_text(pg[j]))
            cos_issue_nm.append(ci_); cos_chain_nm.append(cc_)
            oracle["chain_oracle_pass" if 0.4 * ci_ + 0.6 > 0.7 else "chain_oracle_fail"] += 1
            oracle["issue_oracle_pass" if 0.4 + 0.6 * cc_ > 0.7 else "issue_oracle_fail"] += 1
            if len(examples) < 8 and 0.6 < s <= 0.7:
                examples.append({
                    "sid": sid, "sim": round(float(s), 4),
                    "cos_issue": round(float(ci_), 4), "cos_chain": round(float(cc_), 4),
                    "stance": pst[bi] + "/" + gst[j],
                    "pred_issue": pp[bi]["issue_name"][:40], "gold_issue": pg[j]["issue_name"][:40],
                    "pred_chain0": (pp[bi]["chain"][0][:50] if pp[bi]["chain"] else ""),
                    "gold_chain0": (pg[j]["chain"][0][:50] if pg[j]["chain"] else "")})
        for i in range(Np):
            j = assign_p.get(i)
            if j is not None and sim[i, j] > 0.7:
                P_reason["matched"] += 1; continue
            partners = [y for y in range(Ng) if ok[i, y]]
            if not partners:
                P_reason["1_no_stance_partner"] += 1; continue
            bj = max(partners, key=lambda y: sim[i, y])
            if j is not None: P_reason["3_assigned_below_thr"] += 1
            elif sim[i, bj] > 0.7: P_reason["4_pairing_collision"] += 1
            else: P_reason["2_sim_below"] += 1
    return {
        "tag": tag, "NC": NC, "Np": Np_t, "Ng": Ng_t,
        "gold_unmatched_reasons": dict(G_reason), "pred_unmatched_reasons": dict(P_reason),
        "gold_stance_total": dict(G_stance_total), "gold_stance_blocked": dict(G_stance_blocked),
        "pred_stance_dist": dict(pred_stance_dist),
        "nearmiss_bands": dict(bands), "oracle": dict(oracle),
        "nm_cos_issue_mean": round(float(np.mean(cos_issue_nm)), 4) if cos_issue_nm else None,
        "nm_cos_chain_mean": round(float(np.mean(cos_chain_nm)), 4) if cos_chain_nm else None,
        "nearmiss_examples": examples,
    }

# ---------------- D2/D3 ----------------
def d2d3(gold, pred, docs):
    out = {}
    for recs, tag in ((gold, "gold"), (pred, "pred")):
        item_v = item_t = card_full = card_t = 0
        ipc = collections.Counter(); clen = []
        ratios = []; fut_v = fut_t = 0
        fut_cr = []; fut_len = []; doc_len = []
        for r in recs:
            doc = docs.get(r["sample_id"], "")
            docn = norm_ws(doc); docbg = bigrams(doc)
            doc_len.append(len(docn))
            for it in (r.get("issue_list") or []):
                chain = chain_items(it)
                ipc[len(chain)] += 1; card_t += 1
                if not chain: continue
                vs = [norm_ws(c) in docn for c in chain]
                item_v += sum(vs); item_t += len(vs)
                if all(vs): card_full += 1
                for c, v in zip(chain, vs):
                    clen.append(len(norm_ws(c)))
                    if not v: ratios.append(copy_ratio(c, docbg))
            for fa in (r.get("future_argument") or []):
                fa = str(fa); fut_t += 1
                if norm_ws(fa) in docn: fut_v += 1
                fut_cr.append(copy_ratio(fa, docbg)); fut_len.append(len(norm_ws(fa)))
        out[tag] = {
            "cards": card_t,
            "chain_items_per_card": dict(sorted(ipc.items())),
            "item_verbatim_rate": round(item_v / item_t, 4) if item_t else None,
            "card_fully_verbatim_rate": round(card_full / card_t, 4) if card_t else None,
            "nonverbatim_copy_ratio_p50": round(float(np.median(ratios)), 4) if ratios else None,
            "nonverbatim_count": len(ratios),
            "chain_item_chars_p50": int(np.median(clen)) if clen else None,
            "future_count": fut_t,
            "future_verbatim_rate": round(fut_v / fut_t, 4) if fut_t else None,
            "future_copy_ratio_p50": round(float(np.median(fut_cr)), 4) if fut_cr else None,
            "future_copy_ratio_mean": round(float(np.mean(fut_cr)), 4) if fut_cr else None,
            "future_chars_p50": int(np.median(fut_len)) if fut_len else None,
            "doc_chars_p50": int(np.median(doc_len)) if doc_len else None,
        }
    return out

# ---------------- D4 ----------------
def search_prior_bias(prob_list, prior):
    if not prob_list: return (0.0, 0.0), 1e9
    P = np.array(prob_list)
    tgt = np.array([prior["support"], prior["neutral"], prior["oppose"]])
    best, bd = (0.0, 0.0), 1e9
    for bn in np.arange(-1.0, 1.01, 0.1):
        for bo in np.arange(-0.5, 3.01, 0.1):
            lab = (P * np.exp([0.0, bn, bo])).argmax(1)
            dist = np.array([(lab == k).mean() for k in range(3)])
            d = np.abs(dist - tgt).sum()
            if d < bd: bd, best = d, (float(bn), float(bo))
    return best, bd

class FastMatcher:
    def __init__(self, pred, gold, bge):
        self.g = {r["sample_id"]: r for r in gold}
        self.p = {r["sample_id"]: r for r in pred}
        self.samples = []
        for sid, g in self.g.items():
            p = self.p.get(sid, {"issue_list": []})
            pg = norm_triples(g.get("issue_list")); pp = norm_triples(p.get("issue_list"))
            Np, Ng = len(pp), len(pg)
            sim = np.zeros((Np, Ng))
            for i in range(Np):
                for j in range(Ng):
                    sim[i, j] = (0.4 * bge.cos(pp[i]["issue_name"], pg[j]["issue_name"])
                                 + 0.6 * bge.cos(chain_text(pp[i]), chain_text(pg[j])))
            self.samples.append((sid, Np, Ng, sim,
                                 [t["stance"] for t in pp], [t["stance"] for t in pg]))
    def f1(self, stance_of):
        NC = Np_t = Ng_t = 0
        for sid, Np, Ng, sim, pst, gst in self.samples:
            Np_t += Np; Ng_t += Ng
            if Np == 0 or Ng == 0: continue
            ok = np.zeros((Np, Ng), dtype=bool)
            for i in range(Np):
                ps = stance_of(sid, i, pst[i])
                for j in range(Ng):
                    ok[i, j] = (ps == gst[j])
            mat = sim.copy(); mat[~ok] = -1e9
            ri, ci = linear_sum_assignment(-mat)
            NC += sum(1 for i, j in zip(ri, ci) if ok[i, j] and sim[i, j] > 0.7)
        P = NC / Np_t if Np_t else 0.0
        R = NC / Ng_t if Ng_t else 0.0
        return 2 * P * R / (P + R) if (P + R) else 0.0

def d4(gold, pred, rmap, train, bge):
    pos_counts = collections.defaultdict(collections.Counter)
    tot = collections.Counter()
    for r in train:
        for i, it in enumerate(r.get("issue_list") or []):
            st = (it.get("stance") or "").strip().lower()
            if st in STANCES:
                pos_counts[i][st] += 1; tot[st] += 1
    gp = {s: tot[s] / max(1, sum(tot.values())) for s in STANCES}
    pos_prior = {p: ({s: c / sum(cc.values()) for s, c in cc.items()} if sum(cc.values()) >= 40 else gp)
                 for p, cc in pos_counts.items()}
    bypos = collections.defaultdict(list)
    for r in pred:
        for c in cards_of(rmap, r["sample_id"]):
            bypos[c.get("i", 0)].append([c["p_support"], c["p_neutral"], c["p_oppose"]])
    pos_bias = {p: search_prior_bias(pl, pos_prior.get(p, gp))[0] for p, pl in sorted(bypos.items())}
    fm = FastMatcher(pred, gold, bge)
    def probs(sid, i):
        cs = cards_of(rmap, sid)
        if i >= len(cs): return None
        c = cs[i]
        return np.array([c["p_support"], c["p_neutral"], c["p_oppose"]])
    def biased(b):
        def f(sid, i, st):
            p = probs(sid, i)
            if p is None: return st
            return STANCES[int((p * np.exp(b)).argmax())]
        return f
    def pos_f(sid, i, st):
        p = probs(sid, i)
        if p is None: return st
        bn, bo = pos_bias.get(i, (0.0, 0.0))
        return STANCES[int((p * np.exp([0.0, bn, bo])).argmax())]
    def gated(tau):
        def f(sid, i, st):
            p = probs(sid, i)
            if p is None: return st
            if p.max() >= tau: return STANCES[int(p.argmax())]
            return STANCES[int((p * np.exp(GLOBAL_BIAS)).argmax())]
        return f
    res = {}
    res["recorded"] = round(fm.f1(lambda sid, i, st: st), 4)
    res["global_0.7_-0.5"] = round(fm.f1(biased(GLOBAL_BIAS)), 4)
    res["position_prior"] = round(fm.f1(pos_f), 4)
    for tau in (0.6, 0.7, 0.8, 0.9):
        res["gate_%s" % tau] = round(fm.f1(gated(tau)), 4)
    return res, {"global_prior": {k: round(v, 4) for k, v in gp.items()},
                 "pos_prior": {str(p): {k: round(v, 4) for k, v in pr.items()} for p, pr in sorted(pos_prior.items())},
                 "pos_bias": {str(p): [round(float(x), 3) for x in b] for p, b in pos_bias.items()}}

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="/data/neg_opinion/data/val.jsonl")
    ap.add_argument("--pred", default="/data/neg_opinion/eval/pred_r32_val.jsonl")
    ap.add_argument("--rescore", default="/data/neg_opinion/eval/rescore_r32_val.jsonl")
    ap.add_argument("--train", default="/data/neg_opinion/data/train.jsonl")
    ap.add_argument("--bge", default="/data/neg_opinion/models/bge-small-zh-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="/data/neg_opinion/outputs/diag")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    gold = load(a.gold); pred = load(a.pred)
    rmap = load_map(a.rescore); train = load(a.train)
    docs = {r["sample_id"]: ((r.get("docs") or [{}])[0].get("full_text") or "") for r in gold}
    bge = BGE(a.bge, device=a.device)
    def calibrated(sid, i, st):
        cs = cards_of(rmap, sid)
        if i >= len(cs): return st
        c = cs[i]
        sc = np.array([c["p_support"], c["p_neutral"], c["p_oppose"]]) * np.exp(GLOBAL_BIAS)
        return STANCES[int(sc.argmax())]

    print("== D1 归因 (calibrated, 当前操作点) ==", flush=True)
    d1c = d1(gold, pred, rmap, bge, calibrated, "calibrated")
    print("NC=%(NC)s Np=%(Np)s Ng=%(Ng)s" % d1c)
    print("gold未匹配原因:", d1c["gold_unmatched_reasons"])
    print("pred未匹配原因:", d1c["pred_unmatched_reasons"])
    print("gold立场分布:", d1c["gold_stance_total"], " 被立场卡死的:", d1c["gold_stance_blocked"])
    print("pred立场分布(校准后):", d1c["pred_stance_dist"])
    print("近失带分布:", d1c["nearmiss_bands"])
    print("oracle: ", d1c["oracle"])
    print("近失 cos_issue均值=%s cos_chain均值=%s" % (d1c["nm_cos_issue_mean"], d1c["nm_cos_chain_mean"]))

    print("\n== D1 归因 (recorded, baseline 对照) ==", flush=True)
    d1b = d1(gold, pred, rmap, bge, lambda sid, i, st: st, "recorded")
    print("NC=%(NC)s Np=%(Np)s Ng=%(Ng)s" % d1b)
    print("gold未匹配原因:", d1b["gold_unmatched_reasons"])
    print("gold立场被卡死:", d1b["gold_stance_blocked"])

    print("\n== D2/D3 逐字率/溯源 ==", flush=True)
    dd = d2d3(gold, pred, docs)
    for tag in ("gold", "pred"):
        print(tag, json.dumps(dd[tag], ensure_ascii=False))

    print("\n== D4 ⑥变体 (FastMatcher F1, 无标签先验) ==", flush=True)
    d4res, d4meta = d4(gold, pred, rmap, train, bge)
    print(json.dumps(d4res, ensure_ascii=False))
    print("pos_bias:", json.dumps(d4meta["pos_bias"], ensure_ascii=False))

    # 裁决
    print("\n== 裁决 ==", flush=True)
    gu = d1c["gold_unmatched_reasons"]; Ng = d1c["Ng"]
    print("D1 主因: " + ", ".join(f"{k}={v}({v/Ng:.1%})" for k, v in
          sorted(gu.items(), key=lambda kv: -kv[1]) if k != "matched"))
    g, p = dd["gold"], dd["pred"]
    print(f"D2 gold逐字率={g['item_verbatim_rate']} pred逐字率={p['item_verbatim_rate']} "
          f"pred非逐字copy_p50={p['nonverbatim_copy_ratio_p50']}")
    route = "a(选句回填)" if (g["future_verbatim_rate"] or 0) >= 0.5 or (g["future_copy_ratio_p50"] or 0) >= 0.8 else "b(生成式重训)"
    print(f"D3 gold未来论点 verbatim={g['future_verbatim_rate']} copy_p50={g['future_copy_ratio_p50']} "
          f"-> ⑩ 建议 {route}")
    base_f1 = d4res["global_0.7_-0.5"]
    ok_variants = {k: v for k, v in d4res.items()
                   if k not in ("recorded", "global_0.7_-0.5") and v > base_f1 + 0.006}
    print(f"D4 接受变体(>+0.006): {ok_variants if ok_variants else '无, ⑥到此为止'}")

    json.dump({"D1_calibrated": d1c, "D1_recorded": d1b, "D2D3": dd, "D4": d4res, "D4_meta": d4meta},
              open(f"{a.outdir}/phase0.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nsaved", f"{a.outdir}/phase0.json")

if __name__ == "__main__":
    main()
