#!/usr/bin/env python3
"""Stop only superseded selection/monitor jobs and launch the new L20 tranche."""
import subprocess
from pathlib import Path
import psutil
R=Path('/data/neg_opinion');O=R/'outputs/structure_v1';O.mkdir(parents=True,exist_ok=True)
targets={'scripts/selection_research.sh','src/select_candidates.py','scripts/l20_live_monitor.py'}
for process in psutil.process_iter(['pid','cmdline']):
    if targets.intersection(process.info['cmdline'] or []):
        children=process.children(recursive=True)
        process.terminate()
        for child in children:
            try:child.terminate()
            except psutil.NoSuchProcess:pass
        print('stopped superseded job',process.pid)
for script in ['l20_structure_queue.py','l20_structure_monitor.py']:
    with (O/(script+'.log')).open('a') as log:
        p=subprocess.Popen(['/home/yuanhuilin/miniconda3/envs/YHLin/bin/python','-u','scripts/'+script],cwd=R,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print('launched',script,p.pid)
