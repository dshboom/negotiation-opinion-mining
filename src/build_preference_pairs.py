#!/usr/bin/env python3
"""Build chosen/rejected extraction pairs from real fit-only candidates.

Quality uses the official-style scorer against train-holdout gold. Only
same-prompt, real-generation pairs with a clear gap are kept. Pure format
differences and near-ties are excluded to avoid degenerate DPO signal.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidates', required=True, help='dir with cand_c*.jsonl and reports')
    p.add_argument('--gold', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--min-gap', type=float, default=0.05, help='minimum alpha/composite-normalised gap')
    a = p.parse_args()
    root = Path(a.candidates)
    names = ['c0', 'c1', 'c2', 'c3']
    reports = {}
    for n in names:
        rp = root / f'cand_{n}.report.json'
        if not rp.exists():
            raise SystemExit(f'missing {rp}; score candidates first')
        reports[n] = {r['sample_id']: r for r in json.loads(rp.read_text())['per_sample']}
    raw = {}
    for f in (root / 'records').glob('*.json'):
        rec = json.loads(f.read_text())
        raw[(rec['sample_id'], rec['candidate'])] = rec
    ids = sorted({k[0] for k in raw})
    pairs = []
    skipped = Counter()
    for sid in ids:
        scored = []
        for n in names:
            m = reports[n].get(sid)
            if not m:
                skipped['no_report'] += 1
                continue
            rec = raw.get((sid, n))
            parsed = m['Np'] > 0
            truncated = rec is not None and rec.get('finish_reason') == 'length'
            scored.append(dict(name=n, NC=m['NC'], alpha=m['alpha'], Np=m['Np'], Ng=m['Ng'],
                               parsed=parsed, truncated=truncated, score=m['NC'] + m['alpha']))
        valid = [s for s in scored if s['parsed'] and not s['truncated']]
        if len(valid) < 2:
            skipped['too_few_valid'] += 1
            continue
        valid.sort(key=lambda s: (s['NC'], s['alpha']), reverse=True)
        chosen, rejected = valid[0], valid[-1]
        gap = (chosen['NC'] - rejected['NC']) + (chosen['alpha'] - rejected['alpha'])
        # Reject pure format differences: chosen must genuinely add matches or quality.
        if chosen['NC'] == rejected['NC'] and (chosen['alpha'] - rejected['alpha']) < a.min_gap:
            skipped['near_tie'] += 1
            continue
        if chosen['Np'] == 0 or rejected['Np'] == 0:
            skipped['empty'] += 1
            continue
        # Reconstruct normalised answers from the candidate prediction files.
        pred = {n: {r['sample_id']: r for r in load(root / f'cand_{n}.jsonl')} for n in names}
        def answer(n):
            r = pred[n][sid]
            return json.dumps({'issue_list': r['issue_list']}, ensure_ascii=False)
        ctext, rtext = answer(chosen['name']), answer(rejected['name'])
        c_len = sum(len(x['issue_name']) + len(' '.join(x['argument_chain'])) for x in pred[chosen['name']][sid]['issue_list'])
        r_len = sum(len(x['issue_name']) + len(' '.join(x['argument_chain'])) for x in pred[rejected['name']][sid]['issue_list'])
        if c_len > 0 and r_len > 0 and c_len > 2.5 * r_len:
            skipped['length_bias'] += 1
            continue
        rec0 = raw[(sid, chosen['name'])]
        pairs.append(dict(sample_id=sid,
                          chosen=ctext, rejected=rtext,
                          chosen_candidate=chosen['name'], rejected_candidate=rejected['name'],
                          chosen_NC=chosen['NC'], rejected_NC=rejected['NC'],
                          chosen_alpha=round(chosen['alpha'], 4), rejected_alpha=round(rejected['alpha'], 4),
                          gap=round(gap, 4),
                          system=rec0['messages'][0]['content'],
                          user=rec0['messages'][1]['content']))
    out = Path(a.out)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in pairs), encoding='utf-8')
    tmp.replace(out)
    stats = dict(source_samples=len(ids), pairs=len(pairs), skipped=dict(skipped),
                 avg_gap=round(sum(x['gap'] for x in pairs) / len(pairs), 4) if pairs else 0,
                 policy=dict(min_gap=a.min_gap, forbid_truncation=True, forbid_format_only=True,
                             forbid_length_bias=True))
    atomic(out.with_suffix('.stats.json'), stats)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    main()
