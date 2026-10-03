#!/usr/bin/env python3
"""Cache reference log-probabilities for DPO using a frozen task-SFT adapter.

Reference is the structured extraction SFT adapter (not the raw base), so the
DPO objective is anchored to the actual initial policy. Computed once, then the
reference model can be released before policy training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # noqa: E402
from peft import PeftModel  # noqa: E402
from dpo_common import encode_answer, sequence_logp  # noqa: E402


def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/data/public_models/Qwen3-14B')
    p.add_argument('--ref-adapter', required=True)
    p.add_argument('--pairs', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-len', type=int, default=4096)
    p.add_argument('--qlora', action='store_true', default=True)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=qcfg,
                                                 dtype=torch.bfloat16, device_map={'': 0},
                                                 attn_implementation='sdpa')
    model = PeftModel.from_pretrained(model, a.ref_adapter)
    model.eval()
    rows = load(a.pairs)
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(x)['sample_id'] for x in out.read_text(encoding='utf-8').splitlines() if x.strip()}
    fout = out.open('a', encoding='utf-8')
    kept = 0
    for r in rows:
        if r['sample_id'] in done:
            continue
        ce = encode_answer(tok, r['system'], r['user'], r['chosen'], a.max_len)
        re_ = encode_answer(tok, r['system'], r['user'], r['rejected'], a.max_len)
        if ce is None or re_ is None:
            fout.write(json.dumps({'sample_id': r['sample_id'], 'skipped': 'length'}) + '\n')
            fout.flush()
            continue
        def logp(enc):
            ids = torch.tensor([enc[0]], device='cuda')
            lab = torch.tensor([enc[1]], device='cuda')
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
                return float(sequence_logp(logits, lab)[0].cpu())
        fout.write(json.dumps({'sample_id': r['sample_id'], 'ref_chosen': logp(ce),
                               'ref_rejected': logp(re_),
                               'chosen_sup_tokens': sum(1 for x in ce[1] if x != -100),
                               'rejected_sup_tokens': sum(1 for x in re_[1] if x != -100),
                               'max_len': a.max_len}) + '\n')
        fout.flush()
        kept += 1
    fout.close()
    print('cached', kept)


if __name__ == '__main__':
    main()
