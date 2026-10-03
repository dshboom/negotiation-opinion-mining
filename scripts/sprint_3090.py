#!/usr/bin/env python3
"""GPU0 short training, GPU1 inference/scoring pipeline. Fixed-budget exploration only."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/data/dengsiheng/neg_opinion')
PY = str(ROOT/'.venv-qlora/bin/python')
JOBS = [
    ('S01_baseline', 'data/sft_train.jsonl', 1e-4),
    ('S02_evidence_first', 'data/sft_train_evidence_first.jsonl', 1e-4),
    ('S03_baseline_lr5e5', 'data/sft_train.jsonl', 5e-5),
    ('S04_evidence_first_lr5e5', 'data/sft_train_evidence_first.jsonl', 5e-5),
]


def atomic(path, obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)


def prepare():
    train=[json.loads(x) for x in (ROOT/'data/sft_train.jsonl').read_text().splitlines() if x.strip()]
    for r in train:
        obj=json.loads(r['messages'][-1]['content'])
        obj['issue_list']=[dict(argument_chain=c['argument_chain'],issue_name=c['issue_name'],stance=c['stance']) for c in obj['issue_list']]
        r['messages'][-1]['content']=json.dumps(obj,ensure_ascii=False)
    (ROOT/'data/sft_train_evidence_first.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in train),encoding='utf-8')
    gold=[json.loads(x) for x in (ROOT/'data/val.jsonl').read_text().splitlines() if x.strip()]
    # Fixed hash selection; no picking samples by gold quality or model scores.
    gold=sorted(gold,key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())[:60]
    ids={r['sample_id'] for r in gold}
    sft=[json.loads(x) for x in (ROOT/'data/sft_val.jsonl').read_text().splitlines() if x.strip()]
    for name,rows in [('gold60.jsonl',gold),('sft60.jsonl',[r for r in sft if r['sample_id'] in ids])]:
        (ROOT/'outputs/sprints'/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')


def stage(name, kind, gpu, cmd):
    out=ROOT/'outputs/sprints'/name
    out.mkdir(parents=True,exist_ok=True)
    path=out/(kind+'.state.json')
    if path.exists() and json.loads(path.read_text()).get('status')=='done':
        return True
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    state=dict(status='running',started=time.strftime('%FT%T'),gpu=gpu,command=cmd)
    atomic(path,state)
    with (out/(kind+'.log')).open('a') as log:
        result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    state.update(status='done' if result.returncode==0 else 'failed',returncode=result.returncode,finished=time.strftime('%FT%T'))
    if kind=='predict' and result.returncode==0:
        try:
            pred=[json.loads(x) for x in (out/'pred60.jsonl').read_text().splitlines() if x.strip()]
            ids=[r['sample_id'] for r in pred]
            expected={json.loads(x)['sample_id'] for x in (ROOT/'outputs/sprints/gold60.jsonl').read_text().splitlines() if x.strip()}
            assert len(ids)==len(set(ids)) and set(ids)==expected
        except Exception as e:
            state.update(status='failed',validation_error=str(e))
    atomic(path,state)
    return state['status']=='done'


def train():
    try:
        for name,data,lr in JOBS:
            out=ROOT/'outputs/sprints'/name
            out.mkdir(parents=True,exist_ok=True)
            atomic(out/'config.json',dict(exploratory_only=True,steps=40,grad_accum=16,seed=42,max_len=6144,lora_r=16,lr=lr,quantization='NF4 double-quant BF16',loss='Liger fused linear CE',train_data=data,data_sha256=hashlib.sha256((ROOT/data).read_bytes()).hexdigest()))
            cmd=[PY,'-u','src/train_lora.py','--model','models/Qwen3-14B','--train',data,'--out',str(out/'adapter'),'--qlora','--max-steps','40','--grad-accum','16','--lora-r','16','--lora-alpha','32','--max-len','6144','--lr',str(lr),'--seed','42','--save-steps','20','--logging-steps','2']
            stage(name,'train',0,cmd)
    finally:
        (ROOT/'outputs/sprints/TRAIN_WORKER_FINISHED').write_text(time.strftime('%FT%T'))


def evaluate():
    for name,_,_ in JOBS:
        out=ROOT/'outputs/sprints'/name
        while True:
            state=out/'train.state.json'
            if state.exists():
                status=json.loads(state.read_text()).get('status')
                if status in ('done','failed'):break
            if (ROOT/'outputs/sprints/TRAIN_WORKER_FINISHED').exists():break
            time.sleep(20)
        if not state.exists() or json.loads(state.read_text()).get('status')!='done':continue
        if not stage(name,'predict',1,[PY,'-u','src/infer_qlora.py','--model','models/Qwen3-14B','--data','outputs/sprints/sft60.jsonl','--adapter',str(out/'adapter'),'--out',str(out/'pred60.jsonl'),'--batch-size','1']):continue
        stage(name,'score',1,[PY,'-u','src/eval_scorer.py','--gold','outputs/sprints/gold60.jsonl','--pred',str(out/'pred60.jsonl'),'--bge','models/bge-small-zh-v1.5','--bert','models/bert-base-chinese','--device','cuda','--per-sample','--out',str(out/'report60.json')])


def main():
    os.chdir(ROOT)
    folder=ROOT/'outputs/sprints';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    if (ROOT/'env/MAINTENANCE').exists():return
    assert (ROOT/'env/LONG_SMOKE_DONE').exists(),'long-context smoke not passed'
    prepare()
    marker=folder/'TRAIN_WORKER_FINISHED'
    if marker.exists():marker.rename(folder/('TRAIN_WORKER_FINISHED.previous.'+str(int(time.time()))))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(train),pool.submit(evaluate)]
        for f in futures:f.result()
    reports={name:json.loads((folder/name/'report60.json').read_text())['report'] for name,_,_ in JOBS if (folder/name/'report60.json').exists()}
    atomic(folder/'SUMMARY.json',dict(exploratory_only=True,screening_samples=60,training_steps=40,reports=reports,warning='Short-run screening, not full-val acceptance; no automatic winner claim or test generation.'))


if __name__=='__main__':main()
