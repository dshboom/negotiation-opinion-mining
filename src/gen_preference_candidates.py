#!/usr/bin/env python3
"""Generate multiple real extraction candidates on unseen preference-source samples.

Generator is the fit-only S1 adapter. Input prompts contain documents only; gold
answers are never placed in the prompt and are used later only for pair labelling.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

EXTRACT = '抽取文档观点。先选择逐字证据，再概括议题并判断立场。只输出包含 issue_list 的JSON；卡内顺序为 argument_chain、issue_name、stance。不生成未来表态。'
# Greedy first, then moderate diversity; identical prompt, only sampling differs.
SAMPLES = [dict(name='c0', temperature=0.0, seed=42),
           dict(name='c1', temperature=0.8, seed=101),
           dict(name='c2', temperature=0.8, seed=202),
           dict(name='c3', temperature=0.8, seed=303)]


def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--max-new-tokens', type=int, default=1800)
    a = p.parse_args()
    from transformers import AutoTokenizer
    from infer import extract_json, to_submission
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    out = Path(a.out)
    records = out / 'records'
    records.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path('data/structure_v1/preference_split.json').read_text())
    hold = {r['sample_id']: r for r in load('data/structure_v1/holdout.gold.jsonl')}
    ids = split['source_ids']
    if a.limit:
        ids = ids[:a.limit]
    docs = [dict(sample_id=i, docs=hold[i]['docs']) for i in ids]
    tasks = []
    for r in docs:
        text = '\n'.join(d.get('full_text', '') for d in r['docs'])
        for s in SAMPLES:
            tasks.append(dict(key=f"{r['sample_id']}__{s['name']}", sample_id=r['sample_id'],
                              candidate=s['name'], temperature=s['temperature'], seed=s['seed'],
                              messages=[dict(role='system', content=EXTRACT), dict(role='user', content=text)]))
    config = dict(adapter=a.adapter, adapter_sha256=digest(Path(a.adapter) / 'adapter_model.safetensors'),
                  docs_sha256=digest('data/structure_v1/holdout.gold.jsonl'), samples=SAMPLES,
                  max_new_tokens=a.max_new_tokens, limit=a.limit, script_sha256=digest(__file__))
    if (out / 'manifest.json').exists():
        assert json.loads((out / 'manifest.json').read_text()) == config, 'config changed; new dir required'
    else:
        atomic(out / 'manifest.json', config)
    pending = [t for t in tasks if not (records / (t['key'] + '.json')).exists()]
    if pending:
        tok = AutoTokenizer.from_pretrained('/data/public_models/Qwen3-14B')
        for t in pending:
            ids_tok = tok.apply_chat_template(t['messages'], tokenize=True, add_generation_prompt=True,
                                               enable_thinking=False, return_dict=False)
            if len(ids_tok) + a.max_new_tokens > 8192:
                raise ValueError('context overflow; refusing truncation: ' + t['key'])
            t['ids'] = ids_tok
        engine = LLM(model='/data/public_models/Qwen3-14B', dtype='bfloat16', enable_lora=True,
                     max_lora_rank=32, max_model_len=8192, gpu_memory_utilization=.90)
        adapter = LoRARequest('pref', 1, a.adapter)
        # Group by sampling profile so each request uses its intended temperature.
        for s in SAMPLES:
            group = [t for t in pending if t['candidate'] == s['name']]
            params = SamplingParams(temperature=s['temperature'], max_tokens=a.max_new_tokens,
                                    top_p=0.9 if s['temperature'] > 0 else 1.0, seed=s['seed'])
            for start in range(0, len(group), 16):
                batch = group[start:start + 16]
                outputs = engine.generate([dict(prompt_token_ids=t['ids']) for t in batch], params, lora_request=adapter)
                for t, o in zip(batch, outputs):
                    g = o.outputs[0]
                    atomic(records / (t['key'] + '.json'),
                           dict(key=t['key'], sample_id=t['sample_id'], candidate=t['candidate'],
                                temperature=t['temperature'], seed=t['seed'], raw_text=g.text,
                                token_ids=list(g.token_ids), input_tokens=len(t['ids']),
                                finish_reason=g.finish_reason))
                done = len([f for f in records.glob('*.json')])
                atomic(out / 'progress.json', dict(done=done, total=len(tasks), candidate=s['name']))
                print('saved', done, '/', len(tasks), flush=True)
    # Per-candidate prediction files (parsed) for later quality scoring.
    for s in SAMPLES:
        preds = []
        for r in docs:
            rec = json.loads((records / f"{r['sample_id']}__{s['name']}.json").read_text())
            obj = extract_json(rec['raw_text'])
            sub = to_submission(r['sample_id'], obj, '')
            sub['future_argument'] = []
            preds.append(sub)
        tmp = out / f"cand_{s['name']}.jsonl.tmp"
        tmp.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in preds), encoding='utf-8')
        tmp.replace(out / f"cand_{s['name']}.jsonl")
    atomic(out / 'GENERATION_SUCCESS.json',
           dict(samples=len(docs), candidates=len(SAMPLES), requests=len(tasks),
                raw_dir='records', notes='gold labels not used for generation'))
    print('done', len(docs), 'samples', flush=True)


if __name__ == '__main__':
    main()
