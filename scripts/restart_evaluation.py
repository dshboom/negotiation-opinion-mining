#!/usr/bin/env python3
"""Bounded restart of this evaluation only, preserving original logs."""
import json,subprocess,time
from pathlib import Path
import psutil
R=Path('/data/neg_opinion');E=R/'outputs/structure_v1/evaluation_v1'
targets={'scripts/l20_evaluation_queue.py','src/evaluate_structured.py'}
victims={}
for p in psutil.process_iter(['pid','cmdline']):
    if targets.intersection(p.info['cmdline'] or []):
        for q in [p]+p.children(recursive=True):victims[q.pid]=q
(E/'restart_diagnostic.json').write_text(json.dumps(dict(time=time.strftime('%FT%T%z'),reason='weight copy hung in virtual GPU cuMemcpyHtoDAsync sleep; retry spawn instead of fork',pids=list(victims)),indent=2))
for p in victims.values():
    try:p.terminate()
    except psutil.NoSuchProcess:pass
_,alive=psutil.wait_procs(list(victims.values()),timeout=15)
for p in alive:
    try:p.kill()
    except psutil.NoSuchProcess:pass
psutil.wait_procs(alive,timeout=10)
for name in ['queue.log','generate_joint.log']:
    path=E/name
    if path.exists():path.rename(E/(name+'.before_spawn_'+time.strftime('%H%M%S')))
# There are no generation records yet; replace startup-only manifest for changed script hash.
manifest=E/'joint/manifest.json'
if manifest.exists():
    assert not list((E/'joint/records').glob('*.json'))
    manifest.rename(E/'joint/manifest.before_spawn.json')
with (E/'queue.log').open('a') as log:
    p=subprocess.Popen(['/home/yuanhuilin/miniconda3/envs/YHLin/bin/python','-u','scripts/l20_evaluation_queue.py'],cwd=R,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    print('restarted evaluation',p.pid)
