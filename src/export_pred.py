#!/usr/bin/env python3
"""
把模型逐条输出导出为「给人看」的 CSV + JSON，用于分析模型输出哪里可以提升。

输入：
  --gold   带标签的金标 jsonl（有 issue_list / future_argument / docs）
  --pred   模型预测 jsonl（infer / apply_stance_calib 的提交格式）
输出（--outdir 下）：
  model_outputs_cards.csv     每条「预测卡」一行：模型输出 + 最近金标 + 是否命中 + 失败原因
  model_outputs_samples.csv   每篇文档一行：卡数/命中率概览
  model_outputs.json          完整结构化（模型输出、金标、逐卡对齐）
  summary.json                总体统计

匹配与 eval_scorer 完全一致：立场硬门槛 + 0.4*cos(议题名)+0.6*cos(证据链) + 匈牙利 + 阈值 0.7。

用法：
  python src/export_pred.py --gold data/val.jsonl --pred eval/pred_r32_val.jsonl \
      --outdir outputs/export/raw15 --label "r32 (raw)"
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_scorer import BGE, norm_triples, chain_text, triple_equiv, match_and_extract


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


CARD_COLS = [
    "样本ID", "文档类型", "预测卡#", "议题名(模型)", "立场(模型)", "证据条数", "证据链(模型)",
    "未来论点(模型)", "是否命中", "金标议题", "金标立场", "语义分", "cos_议题名", "cos_证据链",
    "失败原因", "该样本预测卡数", "该样本金标卡数",
]
SAMPLE_COLS = [
    "样本ID", "文档类型", "预测卡数", "金标卡数", "命中数", "命中率", "模型议题名(；分隔)",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--bge", default="/data/neg_opinion/models/bge-small-zh-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--include-docs", action="store_true", help="JSON 里附带原文全文（体积大）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 篇（调试）")
    args = ap.parse_args()

    gold = load_jsonl(args.gold)
    pred = load_jsonl(args.pred)
    if args.limit:
        gold = gold[:args.limit]
    pmap = {p["sample_id"]: p for p in pred}
    print(f"[load] gold={len(gold)} pred={len(pred)}", flush=True)

    bge = BGE(args.bge, device=args.device)
    os.makedirs(args.outdir, exist_ok=True)

    card_rows, sample_rows, full = [], [], []
    tot_nc = tot_np = tot_ng = 0
    reason_count = {}

    for gi, g in enumerate(gold):
        sid = g["sample_id"]
        p = pmap.get(sid, {})
        golds = norm_triples(g.get("issue_list"))
        preds = norm_triples(p.get("issue_list"))
        gf = g.get("future_argument") or []
        pf = p.get("future_argument") or []
        docs = g.get("docs") or []
        doc_type = docs[0].get("doc_type", "") if docs else ""

        NC, Np, Ng, pairs, _ = match_and_extract(preds, golds, bge, args.threshold)
        pair_by_pred = {i: (j, s) for i, j, s in pairs}
        tot_nc += NC; tot_np += Np; tot_ng += Ng

        cards = []
        for i, pr in enumerate(preds):
            if i in pair_by_pred:
                j, s = pair_by_pred[i]
                gg = golds[j]
                ci = bge.cos(pr["issue_name"], gg["issue_name"])
                cc = bge.cos(chain_text(pr), chain_text(gg))
                hit = s > args.threshold
                reason = "命中" if hit else "配对成功但相似度≤0.7"
                gj, gs = gg["issue_name"], gg["stance"]
            else:
                same = [(k, triple_equiv(pr, gg, bge)) for k, gg in enumerate(golds)
                        if gg["stance"] == pr["stance"]]
                if not same:
                    reason, gj, gs, ci, cc, s = "立场无同调", "", "", 0.0, 0.0, 0.0
                else:
                    k, best = max(same, key=lambda x: x[1])
                    gj, gs = golds[k]["issue_name"], golds[k]["stance"]
                    ci = bge.cos(pr["issue_name"], golds[k]["issue_name"])
                    cc = bge.cos(chain_text(pr), chain_text(golds[k]))
                    s = best
                    reason = "相似度不足" if best <= args.threshold else "配对挤占"
                hit = False
            reason_count[reason] = reason_count.get(reason, 0) + 1
            fut = str(pf[i]) if i < len(pf) else ""
            row = {
                "样本ID": sid, "文档类型": doc_type, "预测卡#": i + 1,
                "议题名(模型)": pr["issue_name"], "立场(模型)": pr["stance"],
                "证据条数": len(pr["chain"]), "证据链(模型)": " ｜ ".join(pr["chain"]),
                "未来论点(模型)": fut, "是否命中": "1" if hit else "0",
                "金标议题": gj, "金标立场": gs,
                "语义分": round(s, 4), "cos_议题名": round(ci, 4), "cos_证据链": round(cc, 4),
                "失败原因": reason, "该样本预测卡数": Np, "该样本金标卡数": Ng,
            }
            card_rows.append(row)
            cards.append(row)

        sample_rows.append({
            "样本ID": sid, "文档类型": doc_type, "预测卡数": Np, "金标卡数": Ng,
            "命中数": NC, "命中率": round(NC / max(Np, Ng), 4) if max(Np, Ng) else 0.0,
            "模型议题名(；分隔)": "；".join(x["issue_name"] for x in preds),
        })
        entry = {
            "sample_id": sid, "doc_type": doc_type,
            "model_output": {"issue_list": p.get("issue_list", []) if isinstance(p.get("issue_list"), list) else [],
                             "future_argument": pf},
            "gold": {"issue_list": g.get("issue_list", []), "future_argument": gf},
            "stats": {"NC": NC, "Np": Np, "Ng": Ng},
            "per_card": cards,
        }
        if args.include_docs:
            entry["documents"] = docs
        full.append(entry)
        if (gi + 1) % 50 == 0:
            print(f"  {gi+1}/{len(gold)}", flush=True)

    # 写文件
    def write_csv(path, cols, rows):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    write_csv(os.path.join(args.outdir, "model_outputs_cards.csv"), CARD_COLS, card_rows)
    write_csv(os.path.join(args.outdir, "model_outputs_samples.csv"), SAMPLE_COLS, sample_rows)
    with open(os.path.join(args.outdir, "model_outputs.json"), "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "gold": args.gold, "pred": args.pred,
                   "threshold": args.threshold, "samples": full}, f, ensure_ascii=False, indent=2)

    P = tot_nc / tot_np if tot_np else 0.0
    R = tot_nc / tot_ng if tot_ng else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    summary = {
        "label": args.label, "pred": args.pred,
        "samples": len(gold), "NC": tot_nc, "Np": tot_np, "Ng": tot_ng,
        "precision": round(P, 4), "recall": round(R, 4), "F1": round(F1, 4),
        "pred_cards": len(card_rows), # 卡
        "reason_counts": reason_count,
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved ->", os.path.abspath(args.outdir), flush=True)


if __name__ == "__main__":
    main()
