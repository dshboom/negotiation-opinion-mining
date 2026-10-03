#!/usr/bin/env python3
"""Two predeclared follow-up runs after initial four; no test-set predictions."""
import concurrent.futures
import fcntl
import json
import time
import sprint_3090 as s


def main():
    folder=s.ROOT/'outputs/sprints'
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'followup.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    # Start once initial training is finished, not after slow autoregressive evaluation.
    initial=['S01_baseline','S02_evidence_first','S03_baseline_lr5e5','S04_evidence_first_lr5e5']
    while not all((folder/n/'train.state.json').exists() and json.loads((folder/n/'train.state.json').read_text()).get('status') in ('done','failed') for n in initial):
        time.sleep(60)
    s.JOBS=[('S05_baseline_lr2e4','data/sft_train.jsonl',2e-4),
            ('S06_evidence_first_lr2e4','data/sft_train_evidence_first.jsonl',2e-4)]
    marker=folder/'TRAIN_WORKER_FINISHED'
    if marker.exists():marker.rename(folder/('TRAIN_WORKER_FINISHED.initial.'+str(int(time.time()))))
    def evaluate_after_initial():
        # GPU1 remains exclusively owned by initial evaluator until its queue completes.
        while not (folder/'SUMMARY.json').exists():time.sleep(60)
        s.evaluate()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(s.train),pool.submit(evaluate_after_initial)]
        for f in fs:f.result()
    reports={p.parent.name:json.loads(p.read_text())['report'] for p in folder.glob('S*/report60.json')}
    s.atomic(folder/'FOLLOWUP_SUMMARY.json',{'exploratory_only':True,'reports':reports})


if __name__=='__main__':main()
