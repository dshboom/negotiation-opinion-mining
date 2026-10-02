#!/usr/bin/env python3
"""
三个低成本方向的后处理 (复用已存盘预测; val 只做验收):
 ⑥ 立场 logit 校准:  (a) 先验匹配(无标签/无泄漏)  (b) train-calib 扫 F1 最优偏置
 ⑦ 卡片数 k 扫描 + 置信度排序
 ⑧ 近重复卡片去重 (bge issue_name 相似度)
一次加载 bge/bert, 多方案复用, 全部用官方复刻的 eval_scorer 打分。
"""
import os, sys, json, argparse, copy
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_scorer import BGE, BertScorer, evaluate, norm_triples, chain_text
from scipy.optimize import linear_sum_assignment

STANCES = ["support", "neutral", "oppose"]
GLOBAL_PRIOR = {"support": 0.632, "neutral": 0.342, "oppose": 0.026}
DOCTYPE_PRIOR = {
    "联合声明":     {"support": 0.84, "neutral": 0.13, "oppose": 0.023},
    "政策文件":     {"support": 0.69, "neutral": 0.28, "oppose": 0.032},
    "记者会":       {"support": 0.68, "neutral": 0.24, "oppose": 0.086},
    "国际新闻报道": {"support": 0.43, "neutral": 0.55, "oppose": 0.016},
    "评论文章":     {"support": 0.50, "neutral": 0.43, "oppose": 0.071},
}

def load(p):  return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def load_map(p): return {r["sample_id"]: r for r in load(p)}
def cards_of(rmap, sid): return (rmap.get(sid) or {}).get("cards", [])

# ---------------- 快速抽取 F1 (只算抽取, 不含 bert) ----------------
class FastMatcher:
    def __init__(self, pred, gold, bge):
        self.g = {r["sample_id"]: r for r in gold}
        self.p = {r["sample_id"]: r for r in pred}
        self.samples = []
        for sid, g in self.g.items():
            p = self.p.get(sid, {"issue_list": []})
            pg = norm_triples(g.get("issue_list"))
            pp = norm_triples(p.get("issue_list"))
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
            if Np == 0 or Ng == 0:
                continue
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

# ---------------- 偏置搜索 ----------------
def search_prior_bias(prob_list, prior):
    """prob_list: Nx3; 返回 (bn, bo) 使预测类别分布最接近 prior (无标签)"""
    if not prob_list:
        return 0.0, 0.0
    P = np.array(prob_list)
    tgt = np.array([prior["support"], prior["neutral"], prior["oppose"]])
    best, bd = (0.0, 0.0), 1e9
    for bn in np.arange(-1.0, 1.01, 0.1):
        for bo in np.arange(-0.5, 3.01, 0.1):
            lab = (P * np.exp([0.0, bn, bo])).argmax(1)
            dist = np.array([(lab == k).mean() for k in range(3)])
            d = np.abs(dist - tgt).sum()
            if d < bd:
                bd, best = d, (float(bn), float(bo))
    return best, bd

def apply_bias(pred, rmap, b):
    out = []
    for r in pred:
        cards = cards_of(rmap, r["sample_id"])
        items = []
        for i, it in enumerate(r.get("issue_list") or []):
            ni = copy.deepcopy(it)
            if i < len(cards):
                c = cards[i]
                sc = np.array([c["p_support"], c["p_neutral"], c["p_oppose"]]) * np.exp(np.asarray(b))
                ni["stance"] = STANCES[int(sc.argmax())]
            items.append(ni)
        out.append({"sample_id": r["sample_id"], "issue_list": items,
                    "future_argument": list(r.get("future_argument") or [])})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="/data/neg_opinion/data/val.jsonl")
    ap.add_argument("--pred", default="/data/neg_opinion/eval/pred_r32_val.jsonl")
    ap.add_argument("--rescore", default="/data/neg_opinion/eval/rescore_r32_val.jsonl")
    ap.add_argument("--calib-gold", default="/data/neg_opinion/data/calib_gold.jsonl")
    ap.add_argument("--calib-pred", default="/data/neg_opinion/eval/pred_calib.jsonl")
    ap.add_argument("--calib-rescore", default="/data/neg_opinion/eval/rescore_calib.jsonl")
    ap.add_argument("--bge", default="/data/neg_opinion/models/bge-small-zh-v1.5")
    ap.add_argument("--bert", default="/data/neg_opinion/models/bert-base-chinese")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="/data/neg_opinion/outputs/post")
    ap.add_argument("--tag", default="r32")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    gold = load(args.gold); pred = load(args.pred)
    rmap = load_map(args.rescore) if os.path.exists(args.rescore) else {}
    doctype = {g["sample_id"]: g["docs"][0].get("doc_type", "") for g in gold}
    has_rs = bool(rmap)

    bge = BGE(args.bge, device=args.device)
    bs = BertScorer(args.bert, device=args.device, num_layers=9)
    report = {}
    def ev(tag, pr):
        rep, _ = evaluate(gold, pr, bge, bs)
        report[tag] = rep
        print(f"[{tag:26s}] composite={rep['composite']:.4f} F1={rep['F1']:.4f} "
              f"alpha={rep['alpha']:.4f} extract={rep['extract_score']:.4f} pred={rep['predict_score']:.4f}", flush=True)
        return rep

    ev("baseline", pred)

    # ---------- ⑧ 去重 ----------
    emb = {}
    def nem(t):
        if t not in emb: emb[t] = bge.emb(t)
        return emb[t]
    for thr in [0.85, 0.90, 0.93, 0.96]:
        out = []
        for r in pred:
            sid = r["sample_id"]; cards = cards_of(rmap, sid)
            items = r.get("issue_list") or []; fut = r.get("future_argument") or []
            keep = []
            for i, it in enumerate(items):
                dup = next((k for k in keep
                            if float(np.dot(nem(it["issue_name"]), nem(items[k]["issue_name"]))) > thr), None)
                if dup is None:
                    keep.append(i)
                else:
                    li = cards[i]["logp"] if i < len(cards) else -1e9
                    lk = cards[dup]["logp"] if dup < len(cards) else -1e9
                    if li > lk:
                        keep[keep.index(dup)] = i
            out.append({"sample_id": sid, "issue_list": [items[i] for i in keep],
                        "future_argument": [fut[i] if i < len(fut) else "" for i in keep]})
        ev(f"dedup_{thr}", out)
        if thr == 0.90:
            json.dump(out, open(f"{args.outdir}/dedup_090_{args.tag}.json", "w"), ensure_ascii=False)

    # ---------- ⑦ k 扫描 (置信度排序截断) ----------
    kmax = max([len(r.get("issue_list") or []) for r in pred] + [0])
    for k in [3, 4, 5, 6, 7]:
        if k > kmax and k > 5:
            pass
        out = []
        for r in pred:
            sid = r["sample_id"]; cards = cards_of(rmap, sid)
            items = r.get("issue_list") or []; fut = r.get("future_argument") or []
            order = sorted(range(len(items)),
                           key=lambda i: (cards[i]["logp"] if i < len(cards) else -1e9), reverse=True)[:k]
            order = sorted(order)
            out.append({"sample_id": sid, "issue_list": [items[i] for i in order],
                        "future_argument": [fut[i] if i < len(fut) else "" for i in order]})
        ev(f"k_{k}", out)

    if has_rs:
        # ---------- ⑥a 全局先验匹配 ----------
        allp = [c for r in pred for c in cards_of(rmap, r["sample_id"])]
        plist = [[c["p_support"], c["p_neutral"], c["p_oppose"]] for c in allp]
        (bn, bo), d = search_prior_bias(plist, GLOBAL_PRIOR)
        report["prior_global"] = {"b_neutral": bn, "b_oppose": bo, "L1": d}
        ev("stance_prior_global", apply_bias(pred, rmap, [0.0, bn, bo]))

        # ---------- ⑥a per-doc_type 先验 ----------
        by_dt = {}
        for r in pred:
            dt = doctype.get(r["sample_id"], "")
            for c in cards_of(rmap, r["sample_id"]):
                by_dt.setdefault(dt, []).append([c["p_support"], c["p_neutral"], c["p_oppose"]])
        biases = {}
        for dt, prior in DOCTYPE_PRIOR.items():
            biases[dt], _ = search_prior_bias(by_dt.get(dt, []), prior)
        report["prior_doctype"] = biases
        out = []
        for r in pred:
            dt = doctype.get(r["sample_id"], "")
            bn, bo = biases.get(dt, (0.0, 0.0))
            out.append(apply_bias([r], rmap, [0.0, bn, bo])[0])
        ev("stance_prior_doctype", out)

        # ---------- ⑥b train-calib 上扫 F1 最优 ----------
        if os.path.exists(args.calib_pred) and os.path.exists(args.calib_gold):
            cg = load(args.calib_gold); cp = load(args.calib_pred)
            crmap = load_map(args.calib_rescore) if os.path.exists(args.calib_rescore) else {}
            fm = FastMatcher(cp, cg, bge)
            base_f1 = fm.f1(lambda sid, i, st: st)
            best, bf1 = (0.0, 0.0), base_f1
            for bn2 in np.arange(-1.0, 1.01, 0.1):
                for bo2 in np.arange(-0.5, 3.01, 0.1):
                    def sf(sid, i, st, bn2=bn2, bo2=bo2):
                        cs = cards_of(crmap, sid)
                        if i >= len(cs): return st
                        c = cs[i]
                        sc = np.array([c["p_support"], c["p_neutral"], c["p_oppose"]]) * np.exp([0.0, bn2, bo2])
                        return STANCES[int(sc.argmax())]
                    f = fm.f1(sf)
                    if f > bf1: bf1, best = f, (float(bn2), float(bo2))
            report["calib_f1"] = {"base_f1": base_f1, "best_f1": bf1, "b_neutral": best[0], "b_oppose": best[1]}
            print(f"[calib] base_f1={base_f1:.4f} best_f1={bf1:.4f} bias={best}", flush=True)
            ev("stance_calib_f1", apply_bias(pred, rmap, [0.0, best[0], best[1]]))
        else:
            print("[calib] calib pred/gold not found, skip 6b", flush=True)

    json.dump(report, open(f"{args.outdir}/postprocess_{args.tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n==== SUMMARY ====")
    for k, v in report.items():
        if isinstance(v, dict) and "composite" in v:
            print(f"  {k:26s} composite={v['composite']:.4f} F1={v['F1']:.4f} extract={v['extract_score']:.4f}")
    print("saved", f"{args.outdir}/postprocess_{args.tag}.json")

if __name__ == "__main__":
    main()
