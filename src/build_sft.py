#!/usr/bin/env python3
"""把竞赛 jsonl 转成对话式 SFT 数据 (Qwen chat 模板)。"""
import argparse, json

SYS = ("你是外交/时政长文本的观点挖掘专家。给定一篇文档，你需要："
       "1) 抽取议题列表（第 1 个为主议题，其后为子议题），每个议题给出："
       "议题名 issue_name、立场 stance（只能是 support/oppose/neutral 之一）、"
       "以及来自原文的逐字证据链 argument_chain（必须是原文的子串）；"
       "2) 对每个议题预测未来可能出现的观点 future_argument（与 issue_list 顺序、数量一致，80~120 字）。"
       "严格只输出一个 JSON 对象，不要任何多余文字。")

def build_user(rec):
    d = rec["docs"][0]
    return (f"文档类型：{d.get('doc_type','')}\n发布日期：{d.get('publish_date','')}\n\n"
            f"【文档正文】\n{d.get('full_text','')}\n\n"
            "请抽取议题观点并预测未来表态，按 JSON 输出。")

def build_assistant(rec):
    issues = []
    for it in rec.get("issue_list") or []:
        issues.append({
            "issue_name": it.get("issue_name", ""),
            "stance": it.get("stance", ""),
            "argument_chain": list(it.get("argument_chain") or []),
        })
    obj = {"issue_list": issues, "future_argument": list(rec.get("future_argument") or [])}
    return json.dumps(obj, ensure_ascii=False)

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--labeled", action="store_true", help="有标签(train/val)则生成 assistant 目标")
a = ap.parse_args()

rows = []
for line in open(a.inp, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    rec = json.loads(line)
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": build_user(rec)}]
    if a.labeled:
        msgs.append({"role": "assistant", "content": build_assistant(rec)})
    rows.append({"sample_id": rec["sample_id"], "messages": msgs})

with open(a.out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", a.out, len(rows))
