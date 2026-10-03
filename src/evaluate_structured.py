#!/usr/bin/env python3
"""Resumable gold-free generation stages with raw output saved before parsing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER','0')
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')

EXTRACT='抽取文档观点。先选择逐字证据，再概括议题并判断立场。只输出包含 issue_list 的JSON；卡内顺序为 argument_chain、issue_name、stance。不生成未来表态。'
FUTURE='根据原文与议题卡预测该议题未来可能出现的表态，80至120字。不要编造确定的事实，只输出预测正文。'
def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def atomic(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def write_rows(path,rows):
    tmp=path.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8');tmp.replace(path)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['joint','extraction','future'],required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--adapter',required=True)
    p.add_argument('--extraction')
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    from transformers import AutoTokenizer
    from infer import extract_json,to_submission
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    records=out/'records';records.mkdir(exist_ok=True)
    gold=load('data/val.jsonl')
    # Labels are stripped at this boundary; no downstream prompt builder sees gold answers.
    docs=[dict(sample_id=r['sample_id'],docs=r['docs']) for r in gold]
    del gold
    if a.limit:docs=docs[:a.limit]
    tasks=[];extract={}
    if a.stage=='joint':
        sft={r['sample_id']:r for r in load('data/sft_val.jsonl')}
        for r in docs:
            tasks.append(dict(key=r['sample_id'],sample_id=r['sample_id'],messages=[m for m in sft[r['sample_id']]['messages'] if m['role']!='assistant']))
    elif a.stage=='extraction':
        for r in docs:
            text='\n'.join(d.get('full_text','') for d in r['docs'])
            tasks.append(dict(key=r['sample_id'],sample_id=r['sample_id'],messages=[dict(role='system',content=EXTRACT),dict(role='user',content=text)]))
    else:
        extract={r['sample_id']:r for r in load(a.extraction)}
        assert set(extract)=={r['sample_id'] for r in docs}
        for r in docs:
            text='\n'.join(d.get('full_text','') for d in r['docs'])
            for i,c in enumerate(extract[r['sample_id']]['issue_list']):
                tasks.append(dict(key=r['sample_id']+f'__card{i:02d}',sample_id=r['sample_id'],card_index=i,messages=[dict(role='system',content=FUTURE),dict(role='user',content=text+'\n议题卡：'+json.dumps(c,ensure_ascii=False))]))
    pending=[t for t in tasks if not (records/(t['key']+'.json')).exists()]
    maxnew=256 if a.stage=='future' else 1800
    config=dict(stage=a.stage,adapter=a.adapter,adapter_sha256=digest(Path(a.adapter)/'adapter_model.safetensors'),data_sha256=digest('data/val.jsonl'),sft_prompt_sha256=digest('data/sft_val.jsonl') if a.stage=='joint' else None,extraction_sha256=digest(a.extraction) if a.extraction else None,script_sha256=digest(__file__),temperature=0,max_new_tokens=maxnew,repetition_penalty=1.05,thinking=False,limit=a.limit)
    if (out/'manifest.json').exists():
        assert json.loads((out/'manifest.json').read_text())==config,'configuration changed; use a new run directory'
    else:atomic(out/'manifest.json',config)
    if pending:
        from vllm import LLM,SamplingParams
        from vllm.lora.request import LoRARequest
        tok=AutoTokenizer.from_pretrained('/data/public_models/Qwen3-14B')
        for t in pending:
            t['ids']=tok.apply_chat_template(t['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
            if len(t['ids'])+maxnew>8192:raise ValueError('context overflow, refusing truncation: '+t['key'])
        engine=LLM(model='/data/public_models/Qwen3-14B',dtype='bfloat16',enable_lora=True,max_lora_rank=32,max_model_len=8192,gpu_memory_utilization=.90)
        params=SamplingParams(temperature=0,max_tokens=maxnew,repetition_penalty=1.05,seed=42)
        adapter=LoRARequest(a.stage,1,a.adapter)
        for start in range(0,len(pending),16):
            batch=pending[start:start+16]
            outputs=engine.generate([dict(prompt_token_ids=t['ids']) for t in batch],params,lora_request=adapter)
            for t,o in zip(batch,outputs):
                g=o.outputs[0]
                atomic(records/(t['key']+'.json'),dict(key=t['key'],sample_id=t['sample_id'],card_index=t.get('card_index'),messages=t['messages'],raw_text=g.text,token_ids=list(g.token_ids),input_tokens=len(t['ids']),finish_reason=g.finish_reason,stop_reason=g.stop_reason))
            atomic(out/'progress.json',dict(completed=len(tasks)-len(pending)+min(start+16,len(pending)),total=len(tasks),stage=a.stage))
            print('saved',min(start+16,len(pending)),'/',len(pending),flush=True)
    raw=[json.loads((records/(t['key']+'.json')).read_text()) for t in tasks]
    write_rows(out/'raw_generations.jsonl',raw)
    audit=[];pred=[]
    if a.stage!='future':
        for r in raw:
            obj=extract_json(r['raw_text']);normalized=to_submission(r['sample_id'],obj,'')
            if a.stage=='extraction':normalized['future_argument']=[]
            pred.append(normalized)
            try:json.loads(r['raw_text']);strict=True
            except ValueError:strict=False
            audit.append(dict(key=r['key'],parsed=obj is not None,strict_json=strict,finish_reason=r['finish_reason'],input_tokens=r['input_tokens'],output_tokens=len(r['token_ids'])))
    else:
        fmap={r['key']:r for r in raw}
        for r in docs:
            e=extract[r['sample_id']]
            pred.append(dict(sample_id=r['sample_id'],issue_list=e['issue_list'],future_argument=[fmap[r['sample_id']+f'__card{i:02d}']['raw_text'] for i in range(len(e['issue_list']))]))
        audit=[dict(key=r['key'],finish_reason=r['finish_reason'],input_tokens=r['input_tokens'],output_tokens=len(r['token_ids'])) for r in raw]
    write_rows(out/'generation_audit.jsonl',audit)
    write_rows(out/'pred_before_postprocess.jsonl',pred)
    write_rows(out/'pred_scored.jsonl',pred)
    assert len(pred)==len(docs) and len({r['sample_id'] for r in pred})==len(docs)
    atomic(out/'GENERATION_SUCCESS.json',dict(samples=len(pred),requests=len(tasks),length_finishes=sum(r['finish_reason']=='length' for r in raw),raw_sha256=digest(out/'raw_generations.jsonl'),pred_sha256=digest(out/'pred_scored.jsonl')))
if __name__=='__main__':main()
