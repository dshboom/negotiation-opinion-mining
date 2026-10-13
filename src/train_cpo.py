#!/usr/bin/env python3
"""CPO — Contrastive Preference Optimization (Xu et al., ICML 2024) for extraction.

Reference-free sequence-level preference objective plus a behaviour-cloning (BC)
term on the chosen answer, exactly the CPO form (TRL `cpo_alpha`):

    L = -log σ( β ( lp_w - lp_l ) ) + cpo_alpha * ( -lp_w )

`lp_*` are the log-probabilities of the supervised (completion-only) tokens.
With `--norm mean` they are divided by the number of supervised tokens so the
margin is not dominated by answer length; `--norm sum` follows the original
TRL/paper implementation. No reference model is loaded.

Policy is initialised from the structured extraction SFT adapter, and pairs come
from the holdout-only preference split (same pairs the DPO run used).
"""
import argparse
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,  # noqa: E402
                          TrainingArguments, Trainer)
from peft import PeftModel, prepare_model_for_kbit_training  # noqa: E402
from dpo_common import encode_answer, sequence_logp  # noqa: E402


def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


class PairCollator:
    def __init__(self, pad):
        self.pad = pad

    def _pad(self, seqs, labels):
        L = max(len(s) for s in seqs)
        ids, lab, am = [], [], []
        for s, l in zip(seqs, labels):
            n = L - len(s)
            ids.append(s + [self.pad] * n)
            lab.append(l + [-100] * n)
            am.append([1] * len(s) + [0] * n)
        return ids, lab, am

    def __call__(self, batch):
        c = self._pad([b['c_ids'] for b in batch], [b['c_labels'] for b in batch])
        r = self._pad([b['r_ids'] for b in batch], [b['r_labels'] for b in batch])
        t = lambda x: torch.tensor(x, dtype=torch.long)
        return dict(c_ids=t(c[0]), c_labels=t(c[1]), c_am=t(c[2]),
                    r_ids=t(r[0]), r_labels=t(r[1]), r_am=t(r[2]))


class CPOTrainer(Trainer):
    def __init__(self, *args, beta=0.1, cpo_alpha=1.0, norm='mean', **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.cpo_alpha = cpo_alpha
        self.norm = norm
        self._last = {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        c_out = model(input_ids=inputs['c_ids'], attention_mask=inputs['c_am'])
        r_out = model(input_ids=inputs['r_ids'], attention_mask=inputs['r_am'])
        c_lp = sequence_logp(c_out.logits, inputs['c_labels'])
        r_lp = sequence_logp(r_out.logits, inputs['r_labels'])
        if self.norm == 'mean':
            nw = inputs['c_labels'].ne(-100).sum(1).clamp(min=1)
            nl = inputs['r_labels'].ne(-100).sum(1).clamp(min=1)
            lw, ll = c_lp / nw, r_lp / nl
        else:
            lw, ll = c_lp, r_lp
        margin = self.beta * (lw - ll)
        loss_prefer = -F.logsigmoid(margin).mean()
        loss_nll = -lw.mean()
        loss = loss_prefer + self.cpo_alpha * loss_nll
        with torch.no_grad():
            self._last = dict(margin=float(margin.mean().cpu()),
                              preference_accuracy=float((margin > 0).float().mean().cpu()),
                              loss_prefer=float(loss_prefer.cpu()),
                              nll=float(loss_nll.cpu()),
                              loss=float(loss.cpu()))
        return (loss, c_out) if return_outputs else loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/data/public_models/Qwen3-14B')
    p.add_argument('--init-adapter', required=True)
    p.add_argument('--pairs', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--cpo-alpha', type=float, default=1.0,
                   help='weight of the behaviour-cloning (NLL on chosen) term')
    p.add_argument('--norm', choices=['mean', 'sum'], default='sum')
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--epochs', type=float, default=2.0)
    p.add_argument('--bs', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--max-len', type=int, default=4096)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--save-steps', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pairs = load(a.pairs)
    if a.limit:
        pairs = pairs[:a.limit]
    rows, dropped = [], 0
    for pr in pairs:
        ce = encode_answer(tok, pr['system'], pr['user'], pr['chosen'], a.max_len)
        re_ = encode_answer(tok, pr['system'], pr['user'], pr['rejected'], a.max_len)
        if ce is None or re_ is None:
            dropped += 1
            continue
        rows.append(dict(c_ids=ce[0], c_labels=ce[1], r_ids=re_[0], r_labels=re_[1]))
    print(f'[cpo] usable pairs {len(rows)} (dropped {dropped})', flush=True)
    if len(rows) < 8:
        raise SystemExit('too few CPO pairs after length filtering')

    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=qcfg,
                                                 dtype=torch.bfloat16, device_map={'': 0},
                                                 attn_implementation='sdpa')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    kw = dict(output_dir=a.out, per_device_train_batch_size=a.bs,
              gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs,
              max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type='cosine',
              weight_decay=0.0, max_grad_norm=1.0, bf16=True, gradient_checkpointing=True,
              logging_steps=1, save_steps=a.save_steps, save_total_limit=2, report_to='none',
              dataloader_num_workers=0, remove_unused_columns=False, optim='adamw_torch',
              seed=a.seed)
    if 'warmup_ratio' in inspect.signature(TrainingArguments).parameters:
        kw['warmup_ratio'] = 0.03
    else:
        kw['warmup_steps'] = 0.03
    targs = TrainingArguments(**kw)
    ds = Dataset.from_list(rows)
    trainer = CPOTrainer(model=model, args=targs, train_dataset=ds,
                         data_collator=PairCollator(tok.pad_token_id),
                         beta=a.beta, cpo_alpha=a.cpo_alpha, norm=a.norm,
                         processing_class=tok)
    from transformers.trainer_utils import get_last_checkpoint
    last = get_last_checkpoint(a.out) if os.path.isdir(a.out) else None
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    trainer.save_state()
    Path(a.out, 'cpo_summary.json').write_text(json.dumps(
        dict(pairs=len(rows), dropped=dropped, beta=a.beta, cpo_alpha=a.cpo_alpha,
             norm=a.norm, lr=a.lr, epochs=a.epochs, **trainer._last), indent=2))
    print('[cpo] saved', a.out, flush=True)


if __name__ == '__main__':
    main()
