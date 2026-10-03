#!/usr/bin/env python3
import fcntl,json,os,signal,subprocess,time,traceback
from pathlib import Path
R=Path('/data/neg_opinion');O=R/'outputs/structure_v1';E=O/'evaluation_v1';E.mkdir(parents=True,exist_ok=True)
PY='/home/yuanhuilin/miniconda3/envs/YHLin/bin/python';AC='/data/neg_opinion/.venv-accel/bin/python'
def state(value):
    value['time']=time.strftime('%FT%T%z');p=O/'state.json';tmp=p.with_suffix('.eval.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(p)
def run(name,cmd,done):
    if done.exists():return
    logpath=E/(name+'.log')
    with logpath.open('a') as log:
        child=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        started=time.time()
        while child.poll() is None:
            if time.time()-started>600 and time.time()-logpath.stat().st_mtime>600:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise RuntimeError(f'{name}: log silent >10m; stopped only owned stage process group')
            state(dict(stage=name,status='running',pid=child.pid,log=str(logpath)))
            time.sleep(30)
    if child.returncode:raise RuntimeError(f'{name}: exit {child.returncode}; inspect {logpath}')
    assert done.is_file(), 'stage returned without validated artifact: '+name
def main():
    lock=(E/'queue.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    assert all((O/s/'SUCCESS.json').exists() for s in ['B1_joint','S1_extraction','S1_future'])
    for stage,adapter in [('joint','B1_joint'),('extraction','S1_extraction'),('future','S1_future')]:
        folder=E/stage
        cmd=[AC,'-u','src/evaluate_structured.py','--stage',stage,'--adapter',str(O/adapter/'adapter'),'--out',str(folder)]
        if stage=='future':cmd+=['--extraction',str(E/'extraction/pred_scored.jsonl')]
        run('generate_'+stage,cmd,folder/'GENERATION_SUCCESS.json')
        if stage in ['joint','future']:
            run('score_'+stage,[PY,'-u','src/eval_scorer.py','--gold','data/val.jsonl','--pred',str(folder/'pred_scored.jsonl'),'--device','cuda','--per-sample','--out',str(folder/'report.json')],folder/'report.json')
    reports={k:json.loads((E/k/'report.json').read_text())['report'] for k in ['joint','future']}
    assert all(r['samples']==300 for r in reports.values())
    (E/'SUMMARY.json').write_text(json.dumps(dict(postprocess='none',reports=reports,baseline_comparison='Existing 0.5688 includes calibration; compare B1 vs S1 directly first.'),ensure_ascii=False,indent=2))
    state(dict(status='evaluation_complete',stage='val_reports_ready'))
if __name__=='__main__':
    try:main()
    except Exception as e:
        state(dict(status='failed',stage='evaluation',error=str(e)));traceback.print_exc();raise
