#!/usr/bin/env python3
"""
SFT v2: 字段加权 LoRA —— 在 r32 原配方上, 对诊断出的短板字段的 token 提高 loss 权重:
  issue_name ×w_name (命名口径, D1: 近失对 cos_issue=0.52)
  stance     ×w_stance (D1: 101 张被立场硬门槛卡死)
  future_argument ×w_fut (D3: 纯生成, ROUGE-L 0.217)
其余 (argument_chain 已 99.2% 逐字, JSON 语法) 权重 1。
机制: 在 completion 文本里定位字段值字符区间 -> offset 映射到 token -> 权重张量
     -> 自定义 compute_loss 做加权交叉熵 (prompt/think 仍全 mask)。
每样本校验 tokenization 一致性, 不一致则该样本退化为无加权 (并计数, >10% 则中止)。
"""
import os, sys, json, re, argparse, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer)

RE_NAME = re.compile(r'"issue_name"\s*:\s*"((?:[^"\\]|\\.)*)"')
RE_STANCE = re.compile(r'"stance"\s*:\s*"((?:[^"\\]|\\.)*)"')
RE_STR = re.compile(r'"((?:[^"\\]|\\.)*)"')
RE_FUT_ARR = re.compile(r'"future_argument"\s*:\s*\[')

def field_spans(comp_text):
    """返回 [(s,e,kind)] 字符区间 (comp 坐标)"""
    spans = []
    for m in RE_NAME.finditer(comp_text):
        spans.append((m.start(1), m.end(1), "name"))
    for m in RE_STANCE.finditer(comp_text):
        spans.append((m.start(1), m.end(1), "stance"))
    m = RE_FUT_ARR.search(comp_text)
    if m:
        # 扫描数组闭合 (字符串感知)
        i, depth, in_str, esc = m.end(), 0, False, False
        close = None
        while i < len(comp_text):
            c = comp_text[i]
            if in_str:
                if esc: esc = False
                elif c == "\\": esc = True
                elif c == '"': in_str = False
            else:
                if c == '"': in_str = True
                elif c == "[": depth += 1
                elif c == "]":
                    depth -= 1
                    if depth == 0: close = i; break
            i += 1
        if close:
            for s in RE_STR.finditer(comp_text[m.end():close]):
                spans.append((m.end() + s.start(1), m.end() + s.end(1), "fut"))
    return spans

def make_preprocess_v2(tok, max_len, w_name, w_stance, w_fut, stats):
    WMAP = {"name": w_name, "stance": w_stance, "fut": w_fut}
    def fn(ex):
        msgs = ex["messages"]
        if msgs[-1]["role"] != "assistant":
            return {"input_ids": [], "labels": [], "attention_mask": [], "loss_weights": []}
        p_ids = tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_dict=False)
        f_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                        enable_thinking=False, return_dict=False)
        if f_ids[:len(p_ids)] != p_ids:
            return {"input_ids": [], "labels": [], "attention_mask": [], "loss_weights": []}
        comp_text = msgs[-1]["content"]
        prefix_text = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        full_text = tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=False, enable_thinking=False)
        weights = None
        idx = len(prefix_text)
        if full_text.startswith(prefix_text) and full_text[idx:idx + len(comp_text)] == comp_text:
            enc = tok(full_text, return_offsets_mapping=True, add_special_tokens=False)
            if list(enc["input_ids"]) == list(f_ids):
                off = enc["offset_mapping"]
                c0, c1 = idx, idx + len(comp_text)
                wchars = [1.0] * len(comp_text)
                for s, e, kind in field_spans(comp_text):
                    w = WMAP[kind]
                    for c in range(s, e):
                        wchars[c] = max(wchars[c], w)
                weights = [0.0] * len(p_ids)
                for i in range(len(p_ids), len(f_ids)):
                    a, b = off[i]
                    ca, cb = max(a, c0), min(b, c1)
                    if cb <= ca:
                        weights.append(0.0); continue
                    w = 1.0
                    for c in range(ca - c0, cb - c0):
                        if wchars[c] > w: w = wchars[c]
                    weights.append(w)
        if weights is None:
            stats["fallback"] += 1
            weights = [0.0] * len(p_ids) + [1.0] * (len(f_ids) - len(p_ids))
        labels = [-100] * len(p_ids) + list(f_ids[len(p_ids):])
        return {"input_ids": list(f_ids), "labels": labels,
                "attention_mask": [1] * len(f_ids), "loss_weights": weights}
    return fn

class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id
    def __call__(self, batch):
        L = max(len(x["input_ids"]) for x in batch)
        ids, lab, am, w = [], [], [], []
        for x in batch:
            p = L - len(x["input_ids"])
            ids.append(x["input_ids"] + [self.pad_id] * p)
            lab.append(x["labels"] + [-100] * p)
            am.append(x["attention_mask"] + [0] * p)
            w.append(x["loss_weights"] + [0.0] * p)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(lab, dtype=torch.long),
                "attention_mask": torch.tensor(am, dtype=torch.long),
                "loss_weights": torch.tensor(w, dtype=torch.float)}

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        w = inputs.pop("loss_weights")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits                       # bf16 (B,L,V)
        shift_labels = labels[:, 1:]
        shift_w = w[:, 1:]
        Lm1 = shift_labels.shape[1]
        V = logits.shape[-1]
        num = den = None
        CHUNK = 1024                                  # 分块, 避免全序列 float32 logits 爆显存
        for t0 in range(0, Lm1, CHUNK):
            t1 = min(t0 + CHUNK, Lm1)
            lg = logits[:, t0:t1, :].float().reshape(-1, V)
            lb = shift_labels[:, t0:t1].reshape(-1)
            ww = shift_w[:, t0:t1].reshape(-1)
            m = (lb != -100)
            if not bool(m.any()):
                del lg
                continue
            ce = F.cross_entropy(lg, lb, ignore_index=-100, reduction="none")
            n = (ce * ww * m).sum()
            d = (ww * m).sum()
            num = n if num is None else num + n
            den = d if den is None else den + d
            del lg, ce
        loss = (num / den) if (den is not None and float(den) > 0) else (logits.sum() * 0.0)
        return (loss, outputs) if return_outputs else loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/public_models/Qwen3-14B")
    ap.add_argument("--train", default="/data/neg_opinion/data/sft_train.jsonl")
    ap.add_argument("--out", default="/data/neg_opinion/outputs/qwen3-14b-lora-r32-v2w")
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--w-name", type=float, default=2.0)
    ap.add_argument("--w-stance", type=float, default=3.0)
    ap.add_argument("--w-fut", type=float, default=2.0)
    ap.add_argument("--save-steps", type=int, default=150)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    raw = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
    stats = {"fallback": 0}
    pre = make_preprocess_v2(tok, args.max_len, args.w_name, args.w_stance, args.w_fut, stats)

    if args.selftest:
        for ex in raw[:3]:
            r = pre(ex)
            ws = r["loss_weights"]
            hi = [i for i, x in enumerate(ws) if x > 1.0]
            print(f"[selftest] len={len(r['input_ids'])} 加权token={len(hi)} "
                  f"fallback={stats['fallback']}")
            for i in hi[:6]:
                t = tok.decode([r["input_ids"][i]])
                print(f"   w={ws[i]:.0f} token={t!r}")
            tot_name = sum(1 for i in hi if ws[i] == args.w_name)
            print(f"   命名tok={tot_name} (应≈每卡5-6个)")
        print("fallback率:", stats["fallback"], "/", len(raw[:3]))
        return

    ds = Dataset.from_list(raw)
    before = len(ds)
    ds = ds.map(pre, remove_columns=ds.column_names, desc="tokenize")
    ds = ds.filter(lambda x: len(x["input_ids"]) > 0 and len(x["input_ids"]) <= args.max_len)
    print(f"[data] kept {len(ds)}/{before}, fallback(无加权)样本={stats['fallback']}", flush=True)
    if stats["fallback"] > 0.1 * before:
        raise SystemExit("fallback 率 >10%, 中止 (tokenization 校验失败)")

    print("[model] loading ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    from peft import LoraConfig, get_peft_model
    lconf = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                       lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lconf)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.05,
        weight_decay=0.01, max_grad_norm=1.0,
        bf16=True, tf32=True, gradient_checkpointing=True,
        logging_steps=args.logging_steps, save_steps=args.save_steps,
        save_total_limit=3, report_to="none",
        dataloader_num_workers=2, remove_unused_columns=False, seed=args.seed,
        optim="adamw_torch",
    )
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=ds,
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
