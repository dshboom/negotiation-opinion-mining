#!/usr/bin/env python3
"""SimCKP-style absent-keyphrase contrastive fine-tuning of the future module.

Adaptation of SimCKP (Choi et al., Findings of EMNLP 2023) to a causal LM.
In this task the *future argument* is the "absent keyphrase": a statement that
does not appear in the document but must be predicted from it. SimCKP's core idea
is to observe and score a keyphrase *as a whole* by aligning its phrase-level
representation with the corresponding document, contrastively. We implement the
same idea as an auxiliary objective:

    L = CE(future | document+card) + lam * InfoNCE(doc_repr, future_repr)

where `doc_repr` is the attention-masked mean of the last hidden state over the
prompt (system + document + card) and `future_repr` the same over the generated
future statement; the other examples in the batch act as negatives (symmetric
in-batch InfoNCE). Initialised from the S1_future adapter.

The last hidden state is captured with a forward hook on the LM head so that
gradient checkpointing keeps working and no per-layer states are stored.
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


def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def build_examples(tok, rows, max_len):
    ex, dropped = [], 0
    for r in rows:
        msgs = r['messages']
        if msgs[-1]['role'] != 'assistant':
            dropped += 1
            continue
        p_ids = tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_dict=False)
        f_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                        enable_thinking=False, return_dict=False)
        if f_ids[:len(p_ids)] != p_ids:
            dropped += 1
            continue
        if len(f_ids) > max_len or len(f_ids) <= len(p_ids) + 2:
            dropped += 1
            continue
        labels = [-100] * len(p_ids) + list(f_ids[len(p_ids):])
        ex.append(dict(input_ids=list(f_ids), labels=labels))
    return ex, dropped


class Collator:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, batch):
        L = max(len(x['input_ids']) for x in batch)
        ids, lab, am = [], [], []
        for x in batch:
            n = L - len(x['input_ids'])
            ids.append(x['input_ids'] + [self.pad] * n)
            lab.append(x['labels'] + [-100] * n)
            am.append([1] * len(x['input_ids']) + [0] * n)
        t = lambda z: torch.tensor(z, dtype=torch.long)
        return dict(input_ids=t(ids), attention_mask=t(am), labels=t(lab))


class SimCKPTrainer(Trainer):
    def __init__(self, *args, lam=0.2, tau=0.1, bank_size=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam = lam
        self.tau = tau
        self.bank_size = bank_size
        self._bank = None
        self._cap = {}
        self._last = {}
        head = self.model.get_output_embeddings()
        head.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self._cap['h'] = inputs[0]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._cap.clear()
        out = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                    labels=inputs['labels'])
        lm_loss = out.loss
        h = self._cap.get('h')
        if h is None:
            raise RuntimeError('LM-head hook captured no hidden state')
        mask = inputs['attention_mask'].bool()
        lab = inputs['labels']
        prompt_mask = mask & lab.eq(-100)
        comp_mask = mask & lab.ne(-100)
        denom_p = prompt_mask.sum(1, keepdim=True).clamp(min=1)
        denom_c = comp_mask.sum(1, keepdim=True).clamp(min=1)
        doc = (h * prompt_mask.unsqueeze(-1)).sum(1) / denom_p
        fut = (h * comp_mask.unsqueeze(-1)).sum(1) / denom_c
        doc = doc.float()
        fut = fut.float()
        b = doc.shape[0]
        bank = self._bank
        # Anisotropy fix: LLM last-layer pooled states cluster (pairwise cos ~0.97),
        # which makes InfoNCE sit at chance. Centre by the raw negative-bank mean
        # before L2-normalising so the contrast carries signal.
        if bank is not None and bank.shape[0] >= 16:
            c = bank.mean(0, keepdim=True)
            doc = doc - c
            fut = fut - c
        docn = F.normalize(doc, dim=-1)
        futn = F.normalize(fut, dim=-1)
        cols = [docn @ futn.t() / self.tau]
        if bank is not None and bank.shape[0] > 0:
            bankn = F.normalize(bank - (bank.mean(0, keepdim=True) if bank.shape[0] >= 16 else 0.0), dim=-1)
            cols.append(docn @ bankn.t() / self.tau)
        sim = torch.cat(cols, dim=1) if len(cols) > 1 else cols[0]
        if b >= 2:
            tgt = torch.arange(b, device=sim.device)
            contrast = F.cross_entropy(sim, tgt)
        else:
            contrast = sim.new_zeros(())
        with torch.no_grad():
            fd = fut.detach()
            self._bank = fd if self._bank is None else torch.cat([self._bank, fd], 0)[-self.bank_size:]
            if self.state.global_step % 10 == 0:
                print(f'[simckp-step {self.state.global_step}] lm={float(lm_loss):.3f} '
                      f'contrast={float(contrast):.3f} bank={self._bank.shape[0]}', flush=True)
        loss = lm_loss + self.lam * contrast
        with torch.no_grad():
            if b >= 2:
                acc = float((sim.argmax(1) == torch.arange(b, device=sim.device)).float().mean().cpu())
            else:
                acc = 0.0
            self._last = dict(lm_loss=float(lm_loss.cpu()), contrast=float(contrast.cpu()),
                              loss=float(loss.cpu()), batch_acc=acc, batch=b,
                              bank=int(self._bank.shape[0]))
        return (loss, out) if return_outputs else loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/data/public_models/Qwen3-14B')
    p.add_argument('--train', required=True)
    p.add_argument('--init-adapter', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-len', type=int, default=4096)
    p.add_argument('--epochs', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--lam', type=float, default=0.2, help='weight of the InfoNCE term')
    p.add_argument('--tau', type=float, default=0.1, help='InfoNCE temperature')
    p.add_argument('--bank', type=int, default=64, help='size of the detached negative bank')
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--save-steps', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    rows = load(a.train)
    if a.limit:
        rows = rows[:a.limit]
    ex, dropped = build_examples(tok, rows, a.max_len)
    print(f'[simckp] examples {len(ex)} (dropped {dropped})', flush=True)
    if len(ex) < 8:
        raise SystemExit('too few SimCKP examples')

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
              dataloader_num_workers=2, remove_unused_columns=False, optim='adamw_torch',
              seed=a.seed)
    if 'warmup_ratio' in inspect.signature(TrainingArguments).parameters:
        kw['warmup_ratio'] = 0.03
    else:
        kw['warmup_steps'] = 0.03
    targs = TrainingArguments(**kw)
    ds = Dataset.from_list(ex)
    trainer = SimCKPTrainer(model=model, args=targs, train_dataset=ds,
                            data_collator=Collator(tok.pad_token_id),
                            lam=a.lam, tau=a.tau, bank_size=a.bank, processing_class=tok)
    from transformers.trainer_utils import get_last_checkpoint
    last = get_last_checkpoint(a.out) if os.path.isdir(a.out) else None
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    trainer.save_state()
    Path(a.out, 'simckp_summary.json').write_text(json.dumps(
        dict(examples=len(ex), dropped=dropped, lam=a.lam, tau=a.tau, lr=a.lr,
             epochs=a.epochs, **trainer._last), indent=2))
    print('[simckp] saved', a.out, flush=True)


if __name__ == '__main__':
    main()
