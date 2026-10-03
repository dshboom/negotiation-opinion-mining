#!/usr/bin/env python3
"""Dedicated progress observer and bounded recovery for the dual-3090 queue.

No model API calls, no process termination, no access to the L20 server.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time


def tail(path):
    if not path.exists():
        return ''
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size - 16000))
        return f.read().decode('utf-8', 'replace')


def progress(text):
    generated = re.findall(r'(\d+)/(\d+)\s+ok_json=(\d+)\s+(\d+)s', text)
    if generated:
        done,total,parsed,seconds=generated[-1]
        return {'step':int(done),'total':int(total),'timing':f'{seconds}s elapsed; parsed {parsed}'}
    matches = re.findall(r'\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]', text)
    if not matches:
        return None
    done, total, timing = matches[-1]
    return {'step': int(done), 'total': int(total), 'timing': timing}


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def inspect(root):
    jobs, scores, alerts = {}, {}, []
    sprint_mode = (root/'env/SPRINT_MODE').exists()
    paths = sorted((root/'outputs/sprints').glob('S*/train.state.json')) if sprint_mode else sorted((root/'outputs/overnight').glob('Q*/state.json'))
    for path in paths:
        name = path.parent.name
        try:
            if sprint_mode:
                state = {s:json.loads((path.parent/(s+'.state.json')).read_text()) for s in ['train','predict','score'] if (path.parent/(s+'.state.json')).exists()}
            else:
                state = json.loads(path.read_text())
            info = {}
            for stage, detail in state.items():
                log = path.parent/(stage+'.log')
                text = tail(log)
                info[stage] = {'status': detail.get('status'), 'progress': progress(text)}
                if detail.get('status') == 'running' and log.exists() and time.time()-log.stat().st_mtime > 3600:
                    alerts.append(f'{name}/{stage}: no log update for 1h; observe, do not kill')
                if detail.get('status') == 'failed':
                    kind = ('oom' if 'OutOfMemoryError' in text else
                            'network' if 'ConnectionError' in text or 'ReadTimeout' in text else 'unknown')
                    alerts.append(f'{name}/{stage}: {kind}; retain config, no automatic hyperparameter changes')
            jobs[name] = info
            report = path.parent/('report60.json' if sprint_mode else 'report.json')
            if info.get('score', {}).get('status') == 'done' and report.exists():
                scores[name] = json.loads(report.read_text())['report']
        except (ValueError, OSError, KeyError) as e:
            alerts.append(f'{name}: inspection failed: {e}')
    comparisons = []
    for baseline, evidence in [('Q01_baseline_s42', 'Q02_evidence_first_s42'),
                               ('Q03_baseline_s123', 'Q04_evidence_first_s123')]:
        if baseline in scores and evidence in scores:
            delta = scores[evidence]['composite'] - scores[baseline]['composite']
            comparisons.append({'baseline': baseline, 'variant': evidence, 'composite_delta': delta,
                                'decision': 'candidate improvement; needs seed replication' if delta >= .01 else
                                'no material gain; avoid claiming evidence-first success'})
    return jobs, scores, alerts, comparisons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='/data/dengsiheng/neg_opinion')
    p.add_argument('--interval', type=int, default=300)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    if a.interval < 10:
        p.error('interval must be >=10s')
    root = Path(a.root)
    folder = root/'outputs/monitor3090'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder/'lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    recovery = folder/'recovery.json'
    retries = json.loads(recovery.read_text()) if recovery.exists() else {'queue_restarts': 0}
    previous = None
    while True:
        jobs, scores, alerts, comparisons = inspect(root)
        sprint_mode = (root/'env/SPRINT_MODE').exists()
        script = 'sprint_3090.py' if sprint_mode else 'overnight_3090.py'
        summary = 'outputs/sprints/SUMMARY.json' if sprint_mode else 'outputs/overnight/summary.json'
        running = subprocess.run(['pgrep', '-f', '[p]ython3 scripts/'+script],
                                 stdout=subprocess.DEVNULL).returncode == 0
        paused = (root/'env/MAINTENANCE').exists() or (root/'env/REPAIR_FAILED').exists()
        if not running and (root/'env/SMOKE_DONE').exists() and not paused and not (root/summary).exists():
            if retries['queue_restarts'] < 3:
                with (folder/'queue_recovery.log').open('a') as log:
                    subprocess.Popen(['flock', '-n', 'outputs/overnight.lock', 'python3', 'scripts/'+script],
                                     cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
                retries['queue_restarts'] += 1
                atomic(recovery, retries)
                alerts.append('exited queue: bounded recovery requested')
            else:
                alerts.append('queue recovery limit reached; manual diagnosis required')
        gpu = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],
                             capture_output=True,text=True,timeout=20)
        signature = json.dumps([jobs, scores, alerts, comparisons], sort_keys=True)
        event = signature != previous
        snapshot = {'time':time.strftime('%FT%T%z'), 'jobs':jobs, 'scores':scores,
                    'alerts':alerts, 'comparisons':comparisons, 'gpu':gpu.stdout.strip(),
                    'queue_alive':running, 'maintenance':paused, 'progress_event':event}
        if sprint_mode and (root/'outputs/sprints/SUMMARY.json').exists() and not (root/'outputs/sprints/FOLLOWUP_SUMMARY.json').exists() and not paused:
            active=subprocess.run(['pgrep','-f','[p]ython3 scripts/sprint_followup.py'],stdout=subprocess.DEVNULL).returncode==0
            count=retries.get('followup_restarts',0)
            if not active and count<3:
                with (folder/'followup_recovery.log').open('a') as log:
                    subprocess.Popen(['python3','scripts/sprint_followup.py'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                retries['followup_restarts']=count+1
                atomic(recovery,retries)
        atomic(folder/'latest.json',snapshot)
        if event:
            with (folder/'events.jsonl').open('a') as f:
                f.write(json.dumps(snapshot,ensure_ascii=False)+'\n')
            print(json.dumps(snapshot,ensure_ascii=False),flush=True)
        previous = signature
        if a.once:
            return
        time.sleep(a.interval)


if __name__=='__main__':
    main()
