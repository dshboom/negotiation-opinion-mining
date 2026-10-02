#!/usr/bin/env python3
"""
Qwen3 LoRA / QLoRA SFT —— 遵循 Qwen 官方 (qwen.readthedocs.io MS-SWIFT) 最佳实践:
  * 使用 Qwen3 chat 模板, 训练/推理均 enable_thinking=False
  * 只对 assistant completion 计算 loss (prompt mask 成 -100)
  * 忽略 assistant 开头的空 think 块 '<think>\n\n</think>\n\n' 的 loss
    (= 官方 loss_scale ignore_empty_think)
  * LoRA target all-linear, alpha=2*r, dropout, warmup 5%, bf16, completions-only
"""
import os, sys, json, argparse, time
# NOTE: do NOT use expandable_segments on this HAMi-virtualized GPU (VMM unsupported)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer)

THINK_EMPTY = "<think>\n\n</think>\n\n"

def make_preprocess(tok, max_len):
    def fn(ex):
        msgs = ex["messages"]
        if msgs[-1]["role"] != "assistant":
            return {"input_ids": [], "labels": [], "attention_mask": []}
        # 推理时 prompt (含空 think 块) —— 训练时把这段完全 mask 掉
        p_ids = tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_dict=False)
        f_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                        enable_thinking=False, return_dict=False)
        assert f_ids[:len(p_ids)] == p_ids, "template prefix mismatch"
        labels = [-100] * len(p_ids) + list(f_ids[len(p_ids):])
        return {"input_ids": list(f_ids), "labels": labels,
                "attention_mask": [1] * len(f_ids)}
    return fn

class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id
    def __call__(self, batch):
        L = max(len(x["input_ids"]) for x in batch)
        ids, lab, am = [], [], []
        for x in batch:
            p = L - len(x["input_ids"])
            ids.append(x["input_ids"] + [self.pad_id] * p)
            lab.append(x["labels"] + [-100] * p)
            am.append(x["attention_mask"] + [0] * p)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(lab, dtype=torch.long),
                "attention_mask": torch.tensor(am, dtype=torch.long)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/public_models/Qwen3-14B")
    ap.add_argument("--train", default="/data/neg_opinion/data/sft_train.jsonl")
    ap.add_argument("--out", default="/data/neg_opinion/outputs/qwen3-14b-lora")
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--liger", action="store_true", help="启用 Liger 融合核(需 liger-kernel)")
    ap.add_argument("--save-steps", type=int, default=150)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    raw = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
    ds = Dataset.from_list(raw)
    before = len(ds)
    ds = ds.map(make_preprocess(tok, args.max_len), remove_columns=ds.column_names,
                desc="tokenize")
    ds = ds.filter(lambda x: len(x["input_ids"]) > 0 and len(x["input_ids"]) <= args.max_len)
    print(f"[data] kept {len(ds)}/{before} (max_len={args.max_len})", flush=True)

    print("[model] loading ...", flush=True)
    t0 = time.time()
    load_kw = dict(device_map={"": 0}, attn_implementation="sdpa")
    if args.qlora:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        load_kw["dtype"] = torch.bfloat16
    else:
        load_kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    model.config.use_cache = False
    print(f"[model] loaded in {time.time()-t0:.0f}s, vram {torch.cuda.memory_allocated()/1024**3:.1f}GB", flush=True)

    if args.qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    from peft import LoraConfig, get_peft_model
    lconf = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                       lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
                       target_modules=["q_proj","k_proj","v_proj","o_proj",
                                       "gate_proj","up_proj","down_proj"])
    model = get_peft_model(model, lconf)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.05,
        weight_decay=0.01, max_grad_norm=1.0,
        bf16=True, tf32=True, gradient_checkpointing=True,
        logging_steps=args.logging_steps, save_steps=args.save_steps,
        save_total_limit=3, report_to="none",
        dataloader_num_workers=2, remove_unused_columns=False, seed=args.seed,
        optim="adamw_torch",
        use_liger_kernel=args.liger,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(tok.pad_token_id),
                      processing_class=tok)
    try:
        from transformers.trainer_utils import get_last_checkpoint
        last = get_last_checkpoint(args.out) if os.path.isdir(args.out) else None
    except Exception:
        last = None
    if last:
        print("[resume] from", last, flush=True)
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("[done] saved adapter to", args.out, flush=True)

if __name__ == "__main__":
    main()
