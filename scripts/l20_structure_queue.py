#!/usr/bin/env python3
"""Validated, resumable first SFT tranche; no automatic unvalidated DPO."""
import fcntl,json,subprocess,time,traceback
from pathlib import Path
R=Path('/data/neg_opinion');O=R/'outputs/structure_v1';O.mkdir(parents=True,exist_ok=True)
PY='/home/yuanhuilin/miniconda3/envs/YHLin/bin/python'
def atomic(p,obj):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(p)
def main():
    lock=(O/'queue.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    deadline=time.time()+1800
    while not (O/'preflight.json').exists():
        if time.time()>deadline:raise RuntimeError('preflight not completed within 30m')
        atomic(O/'state.json',dict(status='waiting',stage='preflight',time=time.strftime('%FT%T%z')))
        time.sleep(30)
    pf=json.loads((O/'preflight.json').read_text())
    if pf['train_val_exact_overlap']:raise RuntimeError('train/val duplicates need investigation')
    for stage,task,smoke in [('long_smoke','extraction',True),('S1_extraction','extraction',False),('B1_joint','joint',False),('S1_future','future',False)]:
        folder=O/stage;folder.mkdir(exist_ok=True)
        if (folder/'SUCCESS.json').exists():continue
        data='smoke.'+task if smoke else 'fit.'+task
        maxlen=max(6144,((pf['tasks'][task]['max_tokens']+255)//256)*256)
        cmd=[PY,'-u','src/train_lora.py','--train',f'data/structure_v1/{data}.jsonl','--out',str(folder/'adapter'),'--max-len',str(maxlen),'--strict-data','--epochs','2','--lora-r','32','--lora-alpha','64','--save-steps','25','--logging-steps','1','--grad-accum','16']
        if smoke:cmd+=['--max-steps','1']
        atomic(folder/'config.json',dict(command=cmd,data_sha256=pf['tasks'][task]['sha256'],smoke=smoke,warmup_ratio=.05,thinking=False))
        with (folder/'train.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                progress={}
                checkpoints=list((folder/'adapter').glob('checkpoint-*/trainer_state.json'))
                if checkpoints:
                    p=max(checkpoints,key=lambda p:int(p.parent.name.split('-')[-1]))
                    try:
                        s=json.loads(p.read_text());progress={k:s.get(k) for k in ['global_step','max_steps','epoch']}
                    except ValueError:pass
                atomic(O/'state.json',dict(stage=stage,status='running',pid=child.pid,progress=progress,time=time.strftime('%FT%T%z'),log=str(folder/'train.log')))
                time.sleep(30)
        if child.returncode:raise RuntimeError(f'{stage} failed exit {child.returncode}; no automatic config changes')
        adapter=folder/'adapter'
        assert (adapter/'adapter_model.safetensors').is_file() and (adapter/'trainer_state.json').is_file()
        atomic(folder/'SUCCESS.json',dict(time=time.strftime('%FT%T%z'),stage=stage))
    atomic(O/'state.json',dict(status='sft_complete_pending_inference',stage='next_requires_validated_inference',time=time.strftime('%FT%T%z')))
if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic(O/'state.json',dict(status='failed',error=str(e),time=time.strftime('%FT%T%z')))
        traceback.print_exc();raise
