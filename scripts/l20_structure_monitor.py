#!/usr/bin/env python3
import fcntl,json,subprocess,time
from pathlib import Path
R=Path('/data/neg_opinion');O=R/'outputs/structure_v1';O.mkdir(parents=True,exist_ok=True)
lock=(O/'monitor.lock').open('w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit(0)
while True:
    try:
        state=json.loads((O/'state.json').read_text()) if (O/'state.json').exists() else {'status':'starting'}
        alerts=[]
        if state.get('status')=='failed':alerts.append(state.get('error','queue failed'))
        if state.get('status')=='running':
            log=Path(state['log'])
            silence_limit=600 if state.get('stage','').startswith('generate_') else 1800
            if log.exists() and time.time()-log.stat().st_mtime>silence_limit:alerts.append(f'active log silent >{silence_limit//60}m; possible stall')
            if time.time()-(O/'state.json').stat().st_mtime>180:alerts.append('queue heartbeat stale >3m')
        gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=20).stdout.strip()
        snapshot=dict(time=time.strftime('%FT%T%z'),stage=state.get('stage'),running=state.get('status')=='running',state=state,alerts=alerts,gpu=gpu,tail=Path(state['log']).read_text(errors='replace')[-2000:] if state.get('log') and Path(state['log']).exists() else '')
        p=R/'outputs/night/live.json';tmp=p.with_suffix('.structure.tmp');tmp.write_text(json.dumps(snapshot,ensure_ascii=False));tmp.replace(p)
        (O/'health.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
    except Exception as e:print(repr(e),flush=True)
    time.sleep(60)
