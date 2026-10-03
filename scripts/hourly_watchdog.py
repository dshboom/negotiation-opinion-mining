#!/usr/bin/env python3
"""Hourly persistent health checks; bounded restart of EXITED queues, never kill jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def alive(pattern):
    return subprocess.run(['pgrep', '-f', pattern], stdout=subprocess.DEVNULL).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--role', choices=['3090', 'l20'], required=True)
    a = ap.parse_args()
    root = Path(a.root)
    folder = root/'outputs/monitor'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder/'watchdog.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    statefile = folder/'state.json'
    state = json.loads(statefile.read_text()) if statefile.exists() else {'restarts': {}}
    while True:
        now = time.time()
        alerts = []
        if a.role == '3090':
            tasks = [
                ('setup', '[b]ash scripts/bootstrap_3090.sh', 'env/SMOKE_DONE',
                 ['bash', 'scripts/bootstrap_3090.sh'], 'env/setup.log'),
                ('experiments', '[p]ython3 scripts/overnight_3090.py', 'outputs/overnight/summary.json',
                 ['flock', '-n', 'outputs/overnight.lock', 'python3', 'scripts/overnight_3090.py'], 'outputs/overnight_queue.log'),
            ]
        else:
            # Observe baseline without auto-relaunching legacy scripts with unsafe cleanup.
            tasks = [
                ('baseline', '[b]ash rebuild.sh', 'outputs/REBUILD_DONE', None, 'outputs/rebuild.log'),
                ('export', '[/]bin/bash ./export_when_done.sh', 'outputs/EXPORT_DONE',
                 ['bash', './export_when_done.sh'], 'outputs/export_watchdog.log'),
                ('research', '[b]ash scripts/research_after_rebuild.sh', 'outputs/research/QUEUE_DONE',
                 ['bash', 'scripts/research_after_rebuild.sh'], 'outputs/research/queue.log'),
                ('overnight', '[l]20_overnight_controller.py',
                 'outputs/structure_v1/overnight_v1/FINAL_REPORT.md',
                 ['/home/yuanhuilin/miniconda3/envs/YHLin/bin/python', '-u',
                  'scripts/l20_overnight_controller.py'],
                 'outputs/structure_v1/overnight_v1/watchdog.log'),
                ('preference', '[l]20_preference_queue.py',
                 'outputs/structure_v1/preference_v1/pairs.stats.json',
                 ['/home/yuanhuilin/miniconda3/envs/YHLin/bin/python', '-u',
                  'scripts/l20_preference_queue.py'],
                 'outputs/structure_v1/preference_v1/watchdog.log'),
            ]
        statuses = {}
        for name, pattern, marker, command, logfile in tasks:
            running = alive(pattern)
            complete = (root/marker).exists()
            statuses[name] = {'running': running, 'marker_exists': complete}
            if not complete and not running:
                count = state['restarts'].get(name, 0)
                # Never start GPU experiments while setup has failed.
                permitted = not (root/'env/MAINTENANCE').exists() and not (name == 'experiments' and (root/'env/SETUP_FAILED').exists())
                if command and count < 3 and permitted:
                    log = (root/logfile).open('a')
                    subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
                    log.close()
                    state['restarts'][name] = count + 1
                    alerts.append(f'{name}: exited, restart {count+1}/3 requested')
                else:
                    alerts.append(f'{name}: missing process, manual inspection required')
        # Use active stage logs, not queue stdout (which legitimately stays silent).
        for path in (root/'outputs/overnight').glob('*/state.json'):
            try:
                stages = json.loads(path.read_text())
                for stage, detail in stages.items():
                    if detail.get('status') == 'failed':
                        alerts.append(f'{path.parent.name}/{stage}: failed')
                    if detail.get('status') == 'running':
                        log = path.parent/(stage+'.log')
                        if log.exists() and now-log.stat().st_mtime > 7200:
                            alerts.append(f'{path.parent.name}/{stage}: log silent >2h; NOT killed')
            except (ValueError, OSError) as e:
                alerts.append(f'cannot inspect {path}: {e}')
        if a.role == '3090' and (root/'env/SETUP_FAILED').exists():
            alerts.append('setup/smoke failed: inspect env/setup.log')
        if a.role == 'l20':
            log = root/'outputs/rebuild.log'
            if not (root/'outputs/REBUILD_DONE').exists() and log.exists() and now-log.stat().st_mtime > 7200:
                alerts.append('baseline log silent >2h; NOT killed')
        try:
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu', '--format=csv,noheader'],
                                 capture_output=True, text=True, timeout=20)
            gpu_text = gpu.stdout.strip()
        except Exception as e:
            gpu_text = str(e)
        snapshot = {'time': time.strftime('%FT%T%z'), 'tasks': statuses, 'alerts': alerts,
                    'gpu': gpu_text, 'restarts': state['restarts']}
        tmp = statefile.with_suffix('.tmp')
        tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
        tmp.replace(statefile)
        with (folder/'health.jsonl').open('a') as f:
            f.write(json.dumps(snapshot, ensure_ascii=False)+'\n')
        if alerts:
            (folder/'ALERT.txt').write_text(snapshot['time']+'\n'+'\n'.join(alerts))
        print(json.dumps(snapshot, ensure_ascii=False), flush=True)
        time.sleep(3600)


if __name__ == '__main__':
    main()
