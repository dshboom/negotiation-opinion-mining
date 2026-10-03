#!/usr/bin/env python3
"""Preference-data pipeline for L20: real candidates -> quality scoring -> pairs.

Resumable per stage; unknown failure stops with state. No DPO yet.
"""
import fcntl
import json
import subprocess
import time
import traceback
from pathlib import Path

R = Path('/data/neg_opinion')
O = R / 'outputs/structure_v1'
P = O / 'preference_v1'
P.mkdir(parents=True, exist_ok=True)
PY = '/home/yuanhuilin/miniconda3/envs/YHLin/bin/python'
AC = '/data/neg_opinion/.venv-accel/bin/python'


def state(value):
    value['time'] = time.strftime('%FT%T%z')
    path = O / 'state.json'
    tmp = path.with_suffix('.pref.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def run(name, cmd, done):
    if done.exists():
        return
    logpath = P / (name + '.log')
    with logpath.open('a') as log:
        child = subprocess.Popen(cmd, cwd=R, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.time()
        while child.poll() is None:
            if time.time() - started > 900 and time.time() - logpath.stat().st_mtime > 900:
                raise RuntimeError(f'{name}: log silent >15m; inspect {logpath}')
            state(dict(stage=name, status='running', pid=child.pid, log=str(logpath)))
            time.sleep(30)
    if child.returncode:
        raise RuntimeError(f'{name}: exit {child.returncode}; inspect {logpath}')


def main():
    lock = (P / 'queue.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    assert (O / 'evaluation_v1/SUMMARY.json').exists(), 'finish S1/B1 evaluation first'
    split = json.loads((R / 'data/structure_v1/preference_split.json').read_text())
    source = set(split['source_ids'])
    gold = [json.loads(x) for x in (R / 'data/structure_v1/holdout.gold.jsonl').read_text().splitlines() if x.strip()]
    gold = [g for g in gold if g['sample_id'] in source]
    assert len(gold) == len(source)
    (P / 'gold_source.jsonl').write_text(''.join(json.dumps(g, ensure_ascii=False) + '\n' for g in gold))
    run('generate', [AC, '-u', 'src/gen_preference_candidates.py', '--adapter',
                     str(O / 'S1_extraction/adapter'), '--out', str(P / 'candidates')],
        P / 'candidates/GENERATION_SUCCESS.json')
    for name in ['c0', 'c1', 'c2', 'c3']:
        report = P / 'candidates' / f'cand_{name}.report.json'
        run('score_' + name, [PY, '-u', 'src/eval_scorer.py', '--gold', str(P / 'gold_source.jsonl'),
                              '--pred', str(P / 'candidates' / f'cand_{name}.jsonl'),
                              '--device', 'cuda', '--per-sample', '--out', str(report)], report)
    run('build_pairs', [PY, '-u', 'src/build_preference_pairs.py', '--candidates',
                        str(P / 'candidates'), '--gold', str(P / 'gold_source.jsonl'),
                        '--out', str(P / 'pairs.jsonl')], P / 'pairs.jsonl')
    stats = json.loads((P / 'pairs.stats.json').read_text())
    state(dict(status='preference_data_ready', stage='pairs_built', pairs=stats['pairs'],
               skipped=stats['skipped'], caveat='pairs are model-generated and labelled by train-holdout gold'))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        state(dict(status='failed', stage='preference', error=str(exc)))
        traceback.print_exc()
        raise
