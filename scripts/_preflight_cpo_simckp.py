#!/usr/bin/env python3
"""Preflight for the CPO + SimCKP experiments: versions, data sizes, configs."""
import json
from pathlib import Path

R = Path('/data/neg_opinion')
O = R / 'outputs/structure_v1'


def load(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]


print('=== versions ===')
import torch, transformers, peft, datasets  # noqa: E402
print('torch', torch.__version__, '| transformers', transformers.__version__,
      '| peft', peft.__version__, '| datasets', datasets.__version__)

print('=== preference pairs ===')
pairs = load(O / 'preference_v1/pairs.jsonl')
print('pairs:', len(pairs), '| keys:', sorted(pairs[0].keys()))
print('avg chosen chars', round(sum(len(p['chosen']) for p in pairs) / len(pairs)),
      '| avg rejected chars', round(sum(len(p['rejected']) for p in pairs) / len(pairs)))

print('=== fit.future ===')
fut = load(R / 'data/structure_v1/fit.future.jsonl')
print('fit.future rows:', len(fut), '| keys:', sorted(fut[0].keys()))
kinds = {}
for r in fut:
    kinds[r.get('input_kind', '?')] = kinds.get(r.get('input_kind', '?'), 0) + 1
print('input_kind dist:', kinds)

print('=== holder: gold card names ===')
print('=== S1_future/config.json ===')
print((O / 'S1_future/config.json').read_text(encoding='utf-8'))
print('=== S1_extraction/config.json ===')
print((O / 'S1_extraction/config.json').read_text(encoding='utf-8'))

print('=== token length distribution (fit.future, first 300) ===')
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained('/data/public_models/Qwen3-14B')
lens = []
for r in fut[:300]:
    msgs = r['messages']
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                  enable_thinking=False, return_dict=False)
    lens.append(len(ids))
lens.sort()
n = len(lens)
print('n', n, 'p50', lens[n // 2], 'p90', lens[int(n * 0.9)], 'p95', lens[int(n * 0.95)], 'max', max(lens))

print('=== calibration inputs present? ===')
for p in ['calibration_v1/val_extraction_data.jsonl', 'calibration_v1/s1.cal_extraction.jsonl',
          'S1_future/adapter/adapter_model.safetensors', 'S1_extraction/adapter/adapter_model.safetensors']:
    print(p, (O / p).exists())
