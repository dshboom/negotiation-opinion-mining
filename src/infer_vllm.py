#!/usr/bin/env python3
"""vLLM 推理 (离线, 批量, 可断点续跑)。输出提交格式 + 原始文本备查。"""
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
    ap.add_argument("--raw-out", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1600)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-util", type=float, default=0.95)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out, encoding="utf-8"):
            try: done.add(json.loads(l)["sample_id"])
            except Exception: pass
    rows = [r for r in rows if r["sample_id"] not in done]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[vllm] {len(rows)} to run (done {len(done)})", flush=True)
    if not rows:
        return

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", enable_lora=True, max_lora_rank=32,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_util,
              trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, repetition_penalty=1.05)
    lora = LoRARequest("adapter", 1, args.adapter)

    ids_list = []
    for r in rows:
        msgs = [dict(m) for m in r["messages"] if m["role"] != "assistant"]
        if args.extra:
            for m in reversed(msgs):
                if m["role"] == "user":
                    m["content"] = m["content"].rstrip() + "\n" + args.extra
                    break
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      enable_thinking=False, return_dict=False)
        ids_list.append(ids[:args.max_model_len])
    prompts = [{"prompt_token_ids": ids} for ids in ids_list]
    print(f"[vllm] engine ready {time.time()-t0:.0f}s; generating {len(prompts)} ...", flush=True)
    outs = llm.generate(prompts, sp, lora_request=lora)
    dt = time.time() - t0
    fout = open(args.out, "a", encoding="utf-8")
    fraw = open(args.raw_out, "a", encoding="utf-8") if args.raw_out else None
    ntok = 0
    for r, o in zip(rows, outs):
        txt = o.outputs[0].text
        ntok += len(o.outputs[0].token_ids)
        fout.write(json.dumps(to_submission(r["sample_id"], extract_json(txt), ""), ensure_ascii=False) + "\n")
        if fraw:
            fraw.write(json.dumps({"sample_id": r["sample_id"], "text": txt}, ensure_ascii=False) + "\n")
    fout.close()
    if fraw: fraw.close()
    print(f"[vllm] done {dt:.0f}s, {ntok} tokens, {ntok/dt:.1f} tok/s -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
