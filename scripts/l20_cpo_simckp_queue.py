#!/usr/bin/env python3
"""Blocking, resumable queue: CPO (extraction) + SimCKP-style (future) on L20.

For every stage the file artifact is the completion marker, so the queue can be
restarted at any time. All evaluation reuses the frozen structured pipeline:
  * extraction is generated with the candidate adapter, then ⑥ stance-calibrated
    with the candidate's own stance probabilities + the training stance prior;
  * the future module is regenerated on those cards (S1_future by default);
  * the official-style scorer (word ROUGE-L + BERTScore) reports val 300.
"""
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

R = Path('/data/neg_opinion')
O = R / 'outputs/structure_v1'
C = O / 'calibration_v1'
NEW = O / 'cpo_simckp'
NEW.mkdir(parents=True, exist_ok=True)
PY = '/home/yuanhuilin/miniconda3/envs/YHLin/bin/python'
AC = '/data/neg_opinion/.venv-accel/bin/python'
VAL = R / 'data/val.jsonl'
EXTRACT_RESCORE_DATA = C / 'val_extraction_data.jsonl'
S1_CAL = C / 's1.cal_extraction.jsonl'
STATE = NEW / 'state.json'

# name -> CPO args
CPO_VARIANTS = {
    'cpo_sum': ['--norm', 'sum', '--beta', '0.1', '--cpo-alpha', '1.0'],
    'cpo_mean': ['--norm', 'mean', '--beta', '5.0', '--cpo-alpha', '0.1'],
}
SIMCKP_ARGS = ['--lam', '0.3', '--tau', '0.1', '--lr', '1e-5', '--epochs', '1',
               '--bs', '2', '--grad-accum', '8', '--max-steps', '100', '--bank', '64']


def load(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]


def atomic(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def state(**kw):
    kw['time'] = time.strftime('%FT%T%z')
    atomic(STATE, kw)


def run(name, cmd, done, timeout=4 * 3600, silence=1800):
    if done.exists():
        return
    logpath = NEW / (name + '.log')
    with logpath.open('a') as log:
        child = subprocess.Popen(cmd, cwd=R, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        started = time.time()
        while child.poll() is None:
            if time.time() - started > timeout:
                raise RuntimeError(f'{name}: exceeded {timeout}s')
            last = max(logpath.stat().st_mtime, started)
            if time.time() - last > silence:
                os.killpg(child.pid, 15)
                raise RuntimeError(f'{name}: log silent >{silence}s')
            state(stage=name, status='running', pid=child.pid, log=str(logpath))
            time.sleep(30)
    if child.returncode:
        raise RuntimeError(f'{name}: exit {child.returncode}; inspect {logpath}')


def train_cpo(name, extra):
    ad = NEW / name / 'adapter'
    run('train_' + name, [PY, '-u', 'src/train_cpo.py',
                          '--init-adapter', str(O / 'S1_extraction/adapter'),
                          '--pairs', str(O / 'preference_v1/pairs.jsonl'),
                          '--out', str(ad), '--lr', '5e-6', '--epochs', '2',
                          '--grad-accum', '8', '--max-len', '4096', '--save-steps', '20'] + extra,
        ad / 'adapter_model.safetensors', timeout=3600, silence=900)
    return ad


def train_simckp():
    ad = NEW / 'simckp_future' / 'adapter'
    run('train_simckp', [PY, '-u', 'src/train_simckp.py',
                         '--init-adapter', str(O / 'S1_future/adapter'),
                         '--train', str(R / 'data/structure_v1/fit.future.jsonl'),
                         '--out', str(ad), '--max-len', '2048', '--save-steps', '50'] + SIMCKP_ARGS,
        ad / 'adapter_model.safetensors', timeout=6 * 3600, silence=1200)
    return ad


def eval_extraction(name, adapter):
    ex = NEW / f'{name}_extraction'
    run(f'{name}_extract_gen', [AC, '-u', 'src/evaluate_structured.py', '--stage', 'extraction',
                                '--adapter', str(adapter), '--out', str(ex)],
        ex / 'GENERATION_SUCCESS.json', timeout=2400, silence=1200)
    run(f'{name}_extract_score', [PY, '-u', 'src/eval_scorer.py', '--gold', str(VAL),
                                  '--pred', str(ex / 'pred_scored.jsonl'), '--device', 'cuda',
                                  '--per-sample', '--out', str(ex / 'report.json')],
        ex / 'report.json', timeout=1800)
    return ex


def calibrate(name, adapter, ex):
    rescore = NEW / f'{name}.rescore.jsonl'
    run(f'{name}_rescore', [PY, '-u', 'src/rescore.py', '--adapter', str(adapter),
                            '--data', str(EXTRACT_RESCORE_DATA),
                            '--pred', str(ex / 'pred_scored.jsonl'), '--out', str(rescore)],
        rescore, timeout=2400, silence=1200)
    bias = NEW / f'{name}.bias.json'
    run(f'{name}_fit_bias', [PY, '-u', 'src/fit_bias.py', '--rescore', str(rescore),
                             '--out', str(bias)], bias, timeout=900)
    b = json.loads(bias.read_text())
    cal = NEW / f'{name}.cal_extraction.jsonl'
    run(f'{name}_apply', [PY, '-u', 'src/apply_stance_calib.py',
                          '--pred', str(ex / 'pred_scored.jsonl'), '--rescore', str(rescore),
                          '--b-neutral', str(b['b_neutral']), '--b-oppose', str(b['b_oppose']),
                          '--out', str(cal)], cal, timeout=900)
    rep = NEW / f'{name}.cal_extraction.report.json'
    run(f'{name}_score_extraction', [PY, '-u', 'src/eval_scorer.py', '--gold', str(VAL),
                                     '--pred', str(cal), '--device', 'cuda', '--per-sample',
                                     '--out', str(rep)], rep, timeout=1800)
    return cal, json.loads(rep.read_text())['report']


def eval_full(name, future_adapter, cards_file):
    fu = NEW / f'{name}_future'
    run(f'{name}_future_gen', [AC, '-u', 'src/evaluate_structured.py', '--stage', 'future',
                               '--adapter', str(future_adapter), '--out', str(fu),
                               '--extraction', str(cards_file)],
        fu / 'GENERATION_SUCCESS.json', timeout=5400, silence=1800)
    rep = NEW / f'{name}.full.report.json'
    run(f'{name}_score_full', [PY, '-u', 'src/eval_scorer.py', '--gold', str(VAL),
                               '--pred', str(fu / 'pred_scored.jsonl'), '--device', 'cuda',
                               '--per-sample', '--out', str(rep)], rep, timeout=1800)
    return json.loads(rep.read_text())['report']


def report(results):
    lines = ['# CPO 与 SimCKP（缺席关键短语对比）实验对比', '',
             '口径：val 300，官方评分复刻（词级 ROUGE-L + BERTScore，bge 语义匹配）；'
             '综合 = 0.8×抽取 + 0.2×未来。', '',
             '| 方法 | 抽取(未校准) | 抽取(⑥校准) | 未来分 | 综合 |',
             '|---|---:|---:|---:|---:|']
    base = results.get('baseline')
    if base:
        lines.append(f"| S1 结构化 + ⑥（基线） | — | {base['extraction_cal']['extract_score']:.4f} | "
                     f"{base['full']['predict_score']:.4f} | **{base['full']['composite']:.4f}** |")
    fmt = lambda v: f'{v:.4f}' if isinstance(v, (int, float)) else '—'
    for name, r in results.items():
        if name == 'baseline' or not isinstance(r, dict):
            continue
        ex_unc = (r.get('extraction_unc') or {}).get('extract_score')
        ex_cal = (r.get('extraction_cal') or {}).get('extract_score')
        full = r.get('full') or {}
        lines.append(
            f"| {r.get('label', name)} | {fmt(ex_unc)} | {fmt(ex_cal)} | "
            f"{fmt(full.get('predict_score'))} | **{fmt(full.get('composite'))}** |")
    lines += ['', '方法说明：',
              '- **CPO**（Contrastive Preference Optimization）：无参考模型的序列级对比偏好目标 + chosen 行为克隆项，'
              '微调抽取模块；两个变体分别是论文/TRL 默认（sum, β=0.1, cpo_alpha=1.0）与长度归一化（mean, β=5.0, cpo_alpha=0.1）。',
              '- **SimCKP 风格**：把「未来表态」视为缺席关键短语，用 InfoNCE 对齐（原文+议题卡）表示与未来表态表示，'
              '作为因果语言损失之外的辅助目标，微调未来模块。',
              '', '边界：本地评分器，非官方排名；val 被多轮实验反复查看，结论为探索性；test 未改动。']
    (NEW / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    lock = (NEW / 'queue.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    results = {}
    if (C / 'RESULTS.json').exists():
        cal = json.loads((C / 'RESULTS.json').read_text())
        if 's1' in cal:
            results['baseline'] = dict(label='S1 结构化 + ⑥（基线）',
                                       extraction_cal=cal['s1']['extraction'],
                                       full=cal['s1']['full'])

    # 1. CPO variants (extraction)
    for name, extra in CPO_VARIANTS.items():
        try:
            ad = train_cpo(name, extra)
            ex = eval_extraction(name, ad)
            ex_unc = json.loads((ex / 'report.json').read_text())['report']
            cal_file, ex_cal = calibrate(name, ad, ex)
            full = eval_full(f'{name}_cal', O / 'S1_future/adapter', cal_file)
            results[name] = dict(label=('CPO 抽取（sum,β0.1,α1.0）' if name == 'cpo_sum'
                                        else 'CPO 抽取（mean,β5.0,α0.1）'),
                                 extraction_unc=ex_unc, extraction_cal=ex_cal, full=full,
                                 cal_file=str(cal_file))
            atomic(NEW / 'RESULTS.json', results)
        except Exception as exc:
            results[name] = f'failed: {exc}'
            atomic(NEW / 'RESULTS.json', results)
            state(stage=name, status='failed', error=str(exc))

    # 2. SimCKP future module
    try:
        sk = train_simckp()
        full = eval_full('simckp', sk, S1_CAL)
        results['simckp'] = dict(label='SimCKP 未来模块（接 S1+⑥ 抽取）',
                                 full=full)
        atomic(NEW / 'RESULTS.json', results)
    except Exception as exc:
        results['simckp'] = f'failed: {exc}'
        atomic(NEW / 'RESULTS.json', results)
        state(stage='simckp', status='failed', error=str(exc))

    # 3. Combination: best CPO extraction + SimCKP future (if both available)
    try:
        cpo_ok = {k: v for k, v in results.items()
                  if k in CPO_VARIANTS and isinstance(v, dict)}
        sim_ok = isinstance(results.get('simckp'), dict)
        if cpo_ok and sim_ok and (NEW / 'simckp_future/adapter/adapter_model.safetensors').exists():
            best = max(cpo_ok.items(),
                       key=lambda kv: kv[1]['extraction_cal']['extract_score'])
            bname, br = best
            full = eval_full(f'combo_{bname}', NEW / 'simckp_future/adapter',
                             Path(br['cal_file']))
            results['combo'] = dict(label=f'组合：最佳 CPO 抽取 + SimCKP 未来', full=full)
            atomic(NEW / 'RESULTS.json', results)
    except Exception as exc:
        results['combo'] = f'failed: {exc}'
        atomic(NEW / 'RESULTS.json', results)

    report(results)
    atomic(NEW / 'RESULTS.json', results)
    state(stage='done', status='complete')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        state(stage='fatal', status='failed', error=str(exc))
        traceback.print_exc()
        raise
