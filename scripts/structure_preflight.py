#!/usr/bin/env python3
import hashlib
import inspect
import json
from pathlib import Path
from transformers import AutoTokenizer, TrainingArguments

ROOT=Path('/data/neg_opinion')
OUT=ROOT/'outputs/structure_v1'
OUT.mkdir(parents=True,exist_ok=True)
tok=AutoTokenizer.from_pretrained('/data/public_models/Qwen3-14B')
def load(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
manifest=json.loads((ROOT/'data/structure_v1/split_manifest.json').read_text())
fit=set(manifest['fit_ids']); held=set(manifest['holdout_ids'])
assert len(fit)+len(held)==2400 and len(held)>=400 and not fit & held
original=load(ROOT/'data/sft_train.jsonl')
assert len({r['sample_id'] for r in original})==len(original)
joint=[r for r in original if r['sample_id'] in fit]
assert len(joint)==len(fit)
(ROOT/'data/structure_v1/fit.joint.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in joint))
holdrows=load(ROOT/'data/structure_v1/holdout.gold.jsonl')
groups={}
for r in holdrows:
    key=hashlib.sha256('\n'.join(d.get('full_text','') for d in r['docs']).strip().encode()).hexdigest()
    groups.setdefault(key,[]).append(r['sample_id'])
dev=[];source=[]
for key in sorted(groups):
    (dev if len(dev)<100 else source).extend(groups[key])
(ROOT/'data/structure_v1/preference_split.json').write_text(json.dumps(dict(source_ids=source,dev_ids=dev),indent=2))
docsets={}
for name,path in [('fit','data/structure_v1/fit.gold.jsonl'),('holdout','data/structure_v1/holdout.gold.jsonl'),('val','data/val.jsonl')]:
    hashes=[hashlib.sha256('\n'.join(d.get('full_text','') for d in r['docs']).strip().encode()).hexdigest() for r in load(ROOT/path)]
    docsets[name]=set(hashes)
assert not docsets['fit'] & docsets['holdout'], 'fit/holdout duplicate documents'
val_overlap=len((docsets['fit']|docsets['holdout']) & docsets['val'])
stats={}
for name in ['extraction','future','joint']:
    path=ROOT/('data/structure_v1/fit.'+name+'.jsonl')
    rows=load(path); lengths=[]; supervised=[]; processed=[]
    for r in rows:
        msgs=r['messages']
        assert msgs[-1]['role']=='assistant' and isinstance(msgs[-1]['content'],str)
        prompt=tok.apply_chat_template(msgs[:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        full=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False,enable_thinking=False,return_dict=False)
        assert full[:len(prompt)]==prompt and len(full)>len(prompt)
        lengths.append(len(full));supervised.append(len(full)-len(prompt));processed.append((len(full),r))
    longest=[r for _,r in sorted(processed,key=lambda x:x[0],reverse=True)[:16]]
    (ROOT/('data/structure_v1/smoke.'+name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in longest))
    stats[name]=dict(samples=len(rows),max_tokens=max(lengths),supervised_tokens=sum(supervised),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
sig=inspect.signature(TrainingArguments)
warmup={k:str(sig.parameters[k]) for k in ['warmup_steps','warmup_ratio'] if k in sig.parameters}
report=dict(tasks=stats,warmup_interface=warmup,fit_holdout_exact_overlap=False,train_val_exact_overlap=val_overlap,near_duplicate_check='not yet implemented',fit=len(fit),holdout=len(held),preference_source=len(source),preference_dev=len(dev))
(OUT/'preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
