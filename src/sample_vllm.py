#!/usr/bin/env python3
"""vLLM 多采样: 每条文档 n 个样本 (T>0), 分块断点续跑。
输出: {out}.run0 ... {out}.run{n-1}, 每行 {sample_id, run, issue_list, future_argument}
"""
import os, sys, json, argparse, time
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
from infer import extract_json, to_submission

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/public_models/Qwen3-14B")
    ap.add_argument("--adapter", default="/data/neg_opinion/outputs/qwen3-14b-lora-r32")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=1600)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-util", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--chunk", type=int, default=50)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    done = set()
    if os.path.exists(args.out + ".run0"):
        for l in open(args.out + ".run0", encoding="utf-8"):
            try: done.add(json.loads(l)["sample_id"])
            except Exception: pass
    rows = [r for r in rows if r["sample_id"] not in done]
    print(f"[sample] {len(rows)} to run (done {len(done)}), n={args.n}, T={args.temperature}", flush=True)
    if not rows: return

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", enable_lora=True, max_lora_rank=32,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_util,
              trust_remote_code=True)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, n=args.n,
                        max_tokens=args.max_new_tokens, repetition_penalty=1.05, seed=args.seed)
    lora = LoRARequest("adapter", 1, args.adapter)

    fouts = [open(f"{args.out}.run{k}", "a", encoding="utf-8") for k in range(args.n)]
    ntok = n_ok = 0
    for b in range(0, len(rows), args.chunk):
        batch = rows[b:b + args.chunk]
        prompts = []
        for r in batch:
            msgs = [dict(m) for m in r["messages"] if m["role"] != "assistant"]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False))
        outs = llm.generate(prompts, sp, lora_request=lora)
        for r, o in zip(batch, outs):
            for k, comp in enumerate(o.outputs):
                pred = to_submission(r["sample_id"], extract_json(comp.text), "")
                pred["run"] = k
                fouts[k].write(json.dumps(pred, ensure_ascii=False) + "\n")
                ntok += len(comp.token_ids)
                if pred["issue_list"]: n_ok += 1
        for f in fouts: f.flush()
        print(f"  {min(b+args.chunk, len(rows))}/{len(rows)}  ok={n_ok}  {time.time()-t0:.0f}s", flush=True)
    for f in fouts: f.close()
    print(f"[sample] done {time.time()-t0:.0f}s, {ntok} tokens, ok_cards={n_ok}/{len(rows)*args.n} -> {args.out}.run*", flush=True)

if __name__ == "__main__":
    main()
