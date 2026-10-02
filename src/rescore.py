#!/usr/bin/env python3
"""
前向打分: 复用已存盘的预测, 用一次 forward 拿 token-logprob。
输出每个样本每张卡: stance 三分类概率 + 卡片平均对数概率(置信度)。
无需重新生成; 不做自回归解码, 比推理便宜很多。
"""
import os, sys, json, argparse, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

STANCES = ["support", "neutral", "oppose"]

def build(model, tok, sample):
    msgs = sample["messages"]
    prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prompt_text

def worker(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map={"": 0}, attn_implementation="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    # 三个类别的首 token
    stance_first = {s: tok.encode(s, add_special_tokens=False)[0] for s in STANCES}
    print("[rescore] stance first ids:", stance_first, flush=True)

    data = {json.loads(l)["sample_id"]: json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()}
    preds = [json.loads(l) for l in open(args.pred, encoding="utf-8") if l.strip()]
    done = set()
    mode = "w"
    if os.path.exists(args.out):
        for l in open(args.out, encoding="utf-8"):
            try: done.add(json.loads(l)["sample_id"])
            except Exception: pass
        mode = "a"
    preds = [p for p in preds if p["sample_id"] not in done]
    print(f"[rescore] {len(preds)} to process (done {len(done)})", flush=True)

    fout = open(args.out, mode, encoding="utf-8")
    t0 = time.time()
    KEY = '"stance": "'
    for n, p in enumerate(preds, 1):
        sid = p["sample_id"]
        rec = data.get(sid)
        if rec is None:
            continue
        msgs = [m for m in rec["messages"] if m["role"] != "assistant"]
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assistant_text = json.dumps({"issue_list": p.get("issue_list") or [],
                                     "future_argument": p.get("future_argument") or []}, ensure_ascii=False)
        full_text = prompt_text + assistant_text
        pstart = len(prompt_text)   # assistant 在 full_text 中的起始字符偏移
        enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True,
                  return_tensors="pt", truncation=True, max_length=args.max_len)
        off = enc.pop("offset_mapping")[0].tolist()
        ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            logits = model(**enc.to("cuda")).logits[0]           # [T, V] bf16
        lf = logits.float()
        lp_tok = lf.gather(-1, ids[0].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lf, dim=-1)
        lp_tok = lp_tok.cpu().tolist()

        # 定位每张卡的 stance token 与卡片字符区间
        cards = []
        pos = 0
        for i, card in enumerate(p.get("issue_list") or []):
            j = assistant_text.find(KEY, pos)
            if j == -1: break
            vstart = j + len(KEY)
            vend = assistant_text.find('"', vstart)
            pos = vend + 1
            ti = next((k for k, (a, b) in enumerate(off) if a <= pstart + vstart < b), None)
            # 卡片区间: 从该卡 "issue_name" 开始到下一张卡之前
            cstart = assistant_text.rfind("{", 0, j)
            cend = assistant_text.find("}", vend)
            span_idx = [k for k, (a, b) in enumerate(off) if a < (pstart + cend + 1) and b > (pstart + cstart)]
            mean_lp = sum(lp_tok[k] for k in span_idx) / max(len(span_idx), 1)
            if ti is not None and ti > 0:
                lg = lf[ti - 1]
                sub = torch.tensor([stance_first[s] for s in STANCES], device=lg.device)
                pr = torch.softmax(lg[sub].float(), dim=-1).cpu().tolist()
            else:
                pr = [0.0, 0.0, 0.0]
            cards.append({"i": i, "stance": card.get("stance", ""),
                          "p_support": round(pr[0], 5), "p_neutral": round(pr[1], 5),
                          "p_oppose": round(pr[2], 5), "logp": round(float(mean_lp), 5)})
        fout.write(json.dumps({"sample_id": sid, "doc_type": p.get("_doc_type", ""),
                               "cards": cards}, ensure_ascii=False) + "\n")
        fout.flush()
        if n % 20 == 0:
            print(f"  {n}/{len(preds)}  {time.time()-t0:.0f}s", flush=True)
    fout.close()
    print(f"[done] {time.time()-t0:.0f}s -> {args.out}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/public_models/Qwen3-14B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=6144)
    worker(ap.parse_args())
