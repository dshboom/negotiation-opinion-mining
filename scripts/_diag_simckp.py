#!/usr/bin/env python3
"""Diagnose SimCKP loss: print LM loss and InfoNCE term separately."""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/data/neg_opinion/src')
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # noqa: E402
from peft import PeftModel, prepare_model_for_kbit_training  # noqa: E402
from train_simckp import build_examples  # noqa: E402

R = Path('/data/neg_opinion')
O = R / 'outputs/structure_v1'


def load(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]


tok = AutoTokenizer.from_pretrained('/data/public_models/Qwen3-14B')
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
rows = load(R / 'data/structure_v1/fit.future.jsonl')[:6]
ex, dropped = build_examples(tok, rows, 2048)
print('examples', len(ex), 'dropped', dropped, 'lens', [len(e['input_ids']) for e in ex])

qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained('/data/public_models/Qwen3-14B', quantization_config=qcfg,
                                             dtype=torch.bfloat16, device_map={'': 0},
                                             attn_implementation='sdpa')
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = PeftModel.from_pretrained(model, str(O / 'S1_future/adapter'), is_trainable=True)
model.config.use_cache = False
model.eval()

cap = {}
model.get_output_embeddings().register_forward_hook(lambda m, i, o: cap.__setitem__('h', i[0]))


def pad(batch):
    L = max(len(e['input_ids']) for e in batch)
    ids, lab, am = [], [], []
    for e in batch:
        n = L - len(e['input_ids'])
        ids.append(e['input_ids'] + [tok.pad_token_id] * n)
        lab.append(e['labels'] + [-100] * n)
        am.append([1] * len(e['input_ids']) + [0] * n)
    return (torch.tensor(ids).cuda(), torch.tensor(am).cuda(), torch.tensor(lab).cuda())


for b in range(2, len(ex) + 1, 2):
    batch = ex[:b]
    ids, am, lab = pad(batch)
    cap.clear()
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=am, labels=lab)
    h = cap['h']
    pm = am.bool() & lab.eq(-100)
    cm = am.bool() & lab.ne(-100)
    doc = (h * pm.unsqueeze(-1)).sum(1) / pm.sum(1, keepdim=True)
    fut = (h * cm.unsqueeze(-1)).sum(1) / cm.sum(1, keepdim=True)
    doc = F.normalize(doc.float(), dim=-1)
    fut = F.normalize(fut.float(), dim=-1)
    sim = doc @ fut.t()
    logits = sim / 0.2
    tgt = torch.arange(b, device=logits.device)
    ce = F.cross_entropy(logits, tgt)
    print(f'b={b} lm_loss={float(out.loss.cpu()):.4f} '
          f'contrast(tau=0.2)={float(ce.cpu()):.4f} '
          f'dot_diag={[round(float(sim[i,i]),3) for i in range(b)]} '
          f'dot_offmax={round(float(sim[0][[j for j in range(b) if j!=0]].max()),3) if b>1 else None}')
