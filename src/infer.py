#!/usr/bin/env python3
"""
Qwen3 推理: 用与训练一致的 prompt, 输出结构化 JSON 预测。
支持: base 模型 或 +LoRA adapter (--adapter)。批处理 + 左 padding。
输出: 提交格式 jsonl (sample_id, issue_list, future_argument)。
"""
import os, sys, json, re, argparse, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_json(text):
    if not text:
        return None
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    s = text[i:j + 1]
    try:
        return json.loads(s)
    except Exception:
        # 常见修复: 去尾逗号
        s2 = re.sub(r",\s*([}\]])", r"\1", s)
        try:
            return json.loads(s2)
        except Exception:
            return None

def to_submission(sid, obj, doc_text):
    issues, futures = [], []
    if isinstance(obj, dict):
        for it in obj.get("issue_list") or []:
            if not isinstance(it, dict):
                continue
            chain = it.get("argument_chain") or []
            if isinstance(chain, str):
                chain = [chain]
            issues.append({"issue_name": str(it.get("issue_name", "")).strip(),
                           "stance": str(it.get("stance", "")).strip().lower(),
                           "argument_chain": [str(c).strip() for c in chain if str(c).strip()]})
        fa = obj.get("future_argument") or []
        if isinstance(fa, str):
            fa = [fa]
        futures = [str(x) for x in fa]
    return {"sample_id": sid, "issue_list": issues, "future_argument": futures}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/public_models/Qwen3-14B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1600)
    ap.add_argument("--max-input-len", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--extra", default=None, help="追加到最后一条 user 消息末尾的指令")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map={"": 0}, attn_implementation="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print("[infer] loaded adapter", args.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    rows = rows[args.start: (args.start + args.limit) if args.limit else None]
    done = set()
    mode = "w"
    if os.path.exists(args.out):
        for l in open(args.out, encoding="utf-8"):
            try:
                done.add(json.loads(l)["sample_id"])
            except Exception:
                pass
        mode = "a"
    rows = [r for r in rows if r["sample_id"] not in done]
    print(f"[infer] {len(rows)} remaining (already done {len(done)}), bs={args.batch_size}, T={args.temperature}", flush=True)
    fout = open(args.out, mode, encoding="utf-8")
    n_ok = 0
    t0 = time.time()
    for b in range(0, len(rows), args.batch_size):
        batch = rows[b:b + args.batch_size]
        prompts = []
        for r in batch:
            msgs = [dict(m) for m in r["messages"] if m["role"] != "assistant"]
            if args.extra:
                for m in reversed(msgs):
                    if m["role"] == "user":
                        m["content"] = m["content"].rstrip() + "\n" + args.extra
                        break
            prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False))
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_input_len).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=args.temperature > 0,
                                 temperature=args.temperature if args.temperature > 0 else None,
                                 top_p=args.top_p, top_k=args.top_k,
                                 repetition_penalty=1.05, pad_token_id=tok.pad_token_id)
        in_len = enc["input_ids"].shape[1]
        for k, r in enumerate(batch):
            txt = tok.decode(gen[k][in_len:], skip_special_tokens=True)
            obj = extract_json(txt)
            if obj is not None:
                n_ok += 1
            pred = to_submission(r["sample_id"], obj, "")
            fout.write(json.dumps(pred, ensure_ascii=False) + "\n")
        fout.flush()
        done = min(b + args.batch_size, len(rows))
        print(f"  {done}/{len(rows)}  ok_json={n_ok}  {time.time()-t0:.0f}s", flush=True)
    fout.close()
    print(f"[done] {n_ok}/{len(rows)} parsed, {time.time()-t0:.0f}s -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
