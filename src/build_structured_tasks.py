#!/usr/bin/env python3
"""Deterministic train-only holdout and evidence-first / conditional-future tasks."""
import argparse
import hashlib
import json
from pathlib import Path


def write(path, rows):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--holdout',type=int,default=400)
    p.add_argument('--seed',default='structure-preference-v1')
    a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.train).read_text(encoding='utf-8').splitlines() if x.strip()]
    ids=[r['sample_id'] for r in rows]
    if len(ids)!=len(set(ids)) or not 0<a.holdout<len(rows):raise ValueError('invalid IDs or holdout size')
    # Identical documents must remain together, even when sample IDs differ.
    groups={}
    for r in rows:
        key=hashlib.sha256('\n'.join(d.get('full_text','') for d in r['docs']).strip().encode()).hexdigest()
        groups.setdefault(key,[]).append(r)
    ranked=sorted(groups,key=lambda k:hashlib.sha256((a.seed+':'+k).encode()).hexdigest())
    held=[];fit=[]
    for key in ranked:
        (held if len(held)<a.holdout else fit).extend(groups[key])
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for split,items in [('fit',fit),('holdout',held)]:
        write(out/(split+'.gold.jsonl'),items)
        extraction=[];future=[]
        for r in items:
            text='\n'.join(d.get('full_text','') for d in r['docs'])
            cards=r['issue_list']; futures=r['future_argument']
            if len(cards)!=len(futures):raise ValueError('card/future count mismatch: '+r['sample_id'])
            ordered=[dict(argument_chain=c['argument_chain'],issue_name=c['issue_name'],stance=c['stance']) for c in cards]
            extraction.append({'sample_id':r['sample_id'],'messages':[
                {'role':'system','content':'抽取文档观点。先选择逐字证据，再概括议题并判断立场。只输出包含 issue_list 的JSON；卡内顺序为 argument_chain、issue_name、stance。不生成未来表态。'},
                {'role':'user','content':text},
                {'role':'assistant','content':json.dumps({'issue_list':ordered},ensure_ascii=False)}]})
            for i,(card,target) in enumerate(zip(cards,futures)):
                future.append({'sample_id':r['sample_id']+f'__card{i:02d}','source_sample_id':r['sample_id'],
                    'input_kind':'gold_card_teacher_forcing','messages':[
                    {'role':'system','content':'根据原文与议题卡预测该议题未来可能出现的表态，80至120字。不要编造确定的事实，只输出预测正文。'},
                    {'role':'user','content':text+'\n议题卡：'+json.dumps(card,ensure_ascii=False)},
                    {'role':'assistant','content':target}]})
        write(out/(split+'.extraction.jsonl'),extraction)
        write(out/(split+'.future.jsonl'),future)
    manifest={'seed':a.seed,'input_sha256':hashlib.sha256(Path(a.train).read_bytes()).hexdigest(),
              'split_unit':'exact_document_group','document_groups':len(groups),
              'fit_ids':[r['sample_id'] for r in fit],'holdout_ids':[r['sample_id'] for r in held],
              'warning':'Only fit may train the generator. Holdout gold is for pair labeling, never generator SFT. Future tasks currently use gold cards; predicted-card augmentation remains required.'}
    (out/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'fit={len(fit)} holdout={len(held)}; train-only split ready')


if __name__=='__main__':main()
