#!/usr/bin/env python3
"""Blocking overnight controller: preference data -> analysis -> RFT -> DPO -> reports.

Every stage is resumable and validated by its artifact. Unknown failure stops the
controller with a recorded error instead of silently continuing.
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
N = O / 'overnight_v1'
N.mkdir(parents=True, exist_ok=True)
P = O / 'preference_v1'
EV = O / 'evaluation_v1'
PY = '/home/yuanhuilin/miniconda3/envs/YHLin/bin/python'
AC = '/data/neg_opinion/.venv-accel/bin/python'
START = time.time()
BUDGET = 6.5 * 3600


def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def atomic(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def state(**kw):
    kw['time'] = time.strftime('%FT%T%z')
    kw['elapsed_h'] = round((time.time() - START) / 3600, 2)
    atomic(O / 'state.json', kw)
    atomic(N / 'state.json', kw)


def run(name, cmd, done, timeout=4 * 3600, silence=1800):
    if done.exists():
        return
    logpath = N / (name + '.log')
    with logpath.open('a') as log:
        child = subprocess.Popen(cmd, cwd=R, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.time()
        while child.poll() is None:
            if time.time() - started > timeout:
                raise RuntimeError(f'{name}: exceeded {timeout}s timeout')
            if time.time() - logpath.stat().st_mtime > silence:
                os.killpg(child.pid, 15)
                raise RuntimeError(f'{name}: log silent >{silence}s; stopped owned process group')
            state(stage=name, status='running', pid=child.pid, log=str(logpath))
            time.sleep(30)
    if child.returncode:
        raise RuntimeError(f'{name}: exit {child.returncode}; inspect {logpath}')


def wait_for(path, stage, timeout=None):
    timeout = timeout or BUDGET
    started = time.time()
    while not path.exists():
        if time.time() - started > timeout:
            raise RuntimeError(f'{stage}: timed out waiting for {path}')
        state(stage=stage, status='waiting', waiting_for=str(path))
        time.sleep(60)


def analyze_pairs():
    pairs = load(P / 'pairs.jsonl')
    cand_stats = {}
    for name in ['c0', 'c1', 'c2', 'c3']:
        rp = P / 'candidates' / f'cand_{name}.report.json'
        if rp.exists():
            cand_stats[name] = json.loads(rp.read_text())['report']
    gaps = [p['gap'] for p in pairs]
    nc_gain = [p['chosen_NC'] - p['rejected_NC'] for p in pairs]
    report = dict(
        pairs=len(pairs),
        candidate_reports=cand_stats,
        avg_gap=round(sum(gaps) / len(gaps), 4) if gaps else 0,
        median_gap=sorted(gaps)[len(gaps) // 2] if gaps else 0,
        pairs_with_nc_gain=sum(1 for g in nc_gain if g > 0),
        avg_nc_gain=round(sum(nc_gain) / len(nc_gain), 3) if nc_gain else 0,
        chosen_candidate_dist={},
        rejected_candidate_dist={},
        gate=dict(min_pairs=40, passed=len(pairs) >= 40),
        usage='RFT uses chosen answers; DPO uses chosen/rejected. Gold labels only label training-side pairs.')
    for p in pairs:
        report['chosen_candidate_dist'][p['chosen_candidate']] = report['chosen_candidate_dist'].get(p['chosen_candidate'], 0) + 1
        report['rejected_candidate_dist'][p['rejected_candidate']] = report['rejected_candidate_dist'].get(p['rejected_candidate'], 0) + 1
    atomic(N / 'PREFERENCE_ANALYSIS.json', report)
    lines = ['# 偏好数据详细分析', '', f"- 样本来源：300 条未参与训练的 holdout",
             f"- 有效偏好对：**{len(pairs)}**", f"- 平均质量差：{report['avg_gap']}，中位差：{report['median_gap']}",
             f"- 含真实匹配增益的对：{report['pairs_with_nc_gain']}，平均 NC 增益：{report['avg_nc_gain']}", '',
             '## 四个候选的独立评分', '', '|候选|F1|α|抽取分|NC|', '|---|---:|---:|---:|---:|']
    for name, rep in cand_stats.items():
        lines.append(f"|{name}|{rep['F1']:.4f}|{rep['alpha']:.4f}|{rep['extract_score']:.4f}|{rep['NC']}|")
    lines += ['', '## 入选分布', '', f"- chosen 候选：{report['chosen_candidate_dist']}",
              f"- rejected 候选：{report['rejected_candidate_dist']}", '',
              f"- 质量门槛（≥40 对）：{'通过' if report['gate']['passed'] else '未通过'}",
              '', '## 边界与注意', '',
              '- 候选由 fit-only 模型生成，来源样本未参与其训练。',
              '- 评分为官方口径复刻，用作偏好标签代理，不等于官方排行榜成绩。',
              '- 已排除截断、解析失败、纯格式差异与明显长度偏置的对。',
              '- 偏好数据只用于训练侧；val 300 仍为最终验收。']
    (N / 'PREFERENCE_ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def build_chosen():
    pairs = load(P / 'pairs.jsonl')
    rows = []
    for p in pairs:
        rows.append(dict(sample_id=p['sample_id'], messages=[
            {'role': 'system', 'content': p['system']},
            {'role': 'user', 'content': p['user']},
            {'role': 'assistant', 'content': p['chosen']}]))
    out = N / 'rft_chosen.jsonl'
    out.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf-8')
    print('chosen rows', len(rows))


def evaluate_adapter(tag, adapter):
    ex = EV / f'{tag}_extraction'
    fu = EV / f'{tag}_future'
    run(f'{tag}_extract_gen', [AC, '-u', 'src/evaluate_structured.py', '--stage', 'extraction',
                               '--adapter', str(adapter), '--out', str(ex)],
        ex / 'GENERATION_SUCCESS.json', timeout=3600, silence=1200)
    run(f'{tag}_extract_score', [PY, '-u', 'src/eval_scorer.py', '--gold', 'data/val.jsonl',
                                 '--pred', str(ex / 'pred_scored.jsonl'), '--device', 'cuda',
                                 '--per-sample', '--out', str(ex / 'report.json')],
        ex / 'report.json', timeout=1800)
    run(f'{tag}_future_gen', [AC, '-u', 'src/evaluate_structured.py', '--stage', 'future',
                              '--adapter', str(O / 'S1_future/adapter'), '--out', str(fu),
                              '--extraction', str(ex / 'pred_scored.jsonl')],
        fu / 'GENERATION_SUCCESS.json', timeout=5400, silence=1200)
    run(f'{tag}_full_score', [PY, '-u', 'src/eval_scorer.py', '--gold', 'data/val.jsonl',
                              '--pred', str(fu / 'pred_scored.jsonl'), '--device', 'cuda',
                              '--per-sample', '--out', str(fu / 'report.json')],
        fu / 'report.json', timeout=1800)
    return dict(extraction=json.loads((ex / 'report.json').read_text())['report'],
                full=json.loads((fu / 'report.json').read_text())['report'])


def main():
    lock = (N / 'controller.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    results = {}
    # 1. Block on preference data, then analyse.
    wait_for(P / 'pairs.stats.json', 'waiting_preference_pairs')
    analysis = analyze_pairs()
    results['preference'] = analysis
    if not analysis['gate']['passed']:
        state(status='stopped_low_quality_pairs', stage='analysis', pairs=analysis['pairs'])
        (N / 'FINAL_REPORT.md').write_text(
            f'# 夜间流程\n\n偏好对仅 {analysis["pairs"]} 个，低于门槛，未启动偏好优化。\n', encoding='utf-8')
        return
    build_chosen()
    # 2. RFT: continue SFT from structured extraction on chosen answers.
    rft = N / 'rft_adapter'
    run('rft_train', [PY, '-u', 'src/train_lora.py', '--train', str(N / 'rft_chosen.jsonl'),
                      '--out', str(rft), '--init-adapter', str(O / 'S1_extraction/adapter'),
                      '--max-len', '6656', '--strict-data', '--epochs', '1', '--lr', '5e-5',
                      '--save-steps', '20', '--logging-steps', '1', '--grad-accum', '16'],
        rft / 'trainer_state.json', timeout=3 * 3600)
    results['rft'] = evaluate_adapter('rft', rft)
    atomic(N / 'RESULTS_RFT.json', results['rft'])
    # 3. DPO if budget remains. A DPO failure must not discard the RFT result.
    if time.time() - START < BUDGET - 2.0 * 3600:
        try:
            # Validate the DPO code path on a tiny subset before the full run.
            smoke_pairs = N / 'smoke_pairs.jsonl'
            smoke_pairs.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in load(P / 'pairs.jsonl')[:8]), encoding='utf-8')
            smoke_ref = N / 'smoke_ref.jsonl'
            run('dpo_smoke_cache', [PY, '-u', 'src/cache_ref_logprobs.py', '--ref-adapter',
                                    str(O / 'S1_extraction/adapter'), '--pairs', str(smoke_pairs),
                                    '--out', str(smoke_ref)], smoke_ref, timeout=1800, silence=900)
            smoke_out = N / 'dpo_smoke'
            run('dpo_smoke_train', [PY, '-u', 'src/train_dpo_cached.py', '--init-adapter',
                                    str(O / 'S1_extraction/adapter'), '--pairs', str(smoke_pairs),
                                    '--ref-cache', str(smoke_ref), '--out', str(smoke_out),
                                    '--max-steps', '1', '--save-steps', '1', '--grad-accum', '1'],
                smoke_out / 'dpo_summary.json', timeout=1800, silence=900)
            cache = N / 'ref_logprobs.jsonl'
            run('dpo_cache', [PY, '-u', 'src/cache_ref_logprobs.py', '--ref-adapter',
                              str(O / 'S1_extraction/adapter'), '--pairs', str(P / 'pairs.jsonl'),
                              '--out', str(cache)], cache, timeout=3600, silence=1200)
            dpo = N / 'dpo_adapter'
            run('dpo_train', [PY, '-u', 'src/train_dpo_cached.py', '--init-adapter',
                              str(O / 'S1_extraction/adapter'), '--pairs', str(P / 'pairs.jsonl'),
                              '--ref-cache', str(cache), '--out', str(dpo), '--beta', '0.1',
                              '--lr', '5e-6', '--epochs', '1', '--grad-accum', '8',
                              '--max-len', '4096', '--save-steps', '10'],
                dpo / 'dpo_summary.json', timeout=3 * 3600, silence=1800)
            results['dpo'] = evaluate_adapter('dpo', dpo)
            atomic(N / 'RESULTS_DPO.json', results['dpo'])
        except Exception as exc:
            results['dpo'] = f'failed: {exc}'
            state(stage='dpo_failed', status='running', error=str(exc))
    else:
        results['dpo'] = 'skipped: insufficient remaining budget'
    # 4. Final report.
    cmp = dict(
        baseline=json.loads((EV / 'SUMMARY.json').read_text())['reports'],
        rft=results.get('rft'), dpo=results.get('dpo'),
        preference=dict(pairs=analysis['pairs'], avg_gap=analysis['avg_gap']))
    atomic(N / 'RESULTS_ALL.json', cmp)
    lines = ['# 夜间偏好优化结果', '', f"- 偏好对：{analysis['pairs']}，平均质量差 {analysis['avg_gap']}", '']
    if isinstance(results.get('rft'), dict):
        lines += ['## RFT（用高质量候选继续 SFT）', '',
                  f"- 抽取：F1 {results['rft']['extraction']['F1']:.4f}，α {results['rft']['extraction']['alpha']:.4f}，抽取分 {results['rft']['extraction']['extract_score']:.4f}",
                  f"- 完整系统综合：**{results['rft']['full']['composite']:.4f}**（抽取 {results['rft']['full']['extract_score']:.4f}，未来 {results['rft']['full']['predict_score']:.4f}）", '']
    if isinstance(results.get('dpo'), dict):
        lines += ['## DPO（cached-reference）', '',
                  f"- 抽取：F1 {results['dpo']['extraction']['F1']:.4f}，α {results['dpo']['extraction']['alpha']:.4f}，抽取分 {results['dpo']['extraction']['extract_score']:.4f}",
                  f"- 完整系统综合：**{results['dpo']['full']['composite']:.4f}**", '']
    else:
        lines += ['## DPO', '', f"- {results.get('dpo')}", '']
    lines += ['## 对照（本轮同口径）', '',
              '- B1 联合（fit 2000，无校准）综合 0.5469',
              '- S1 结构化（fit 2000，无校准）综合 0.5652',
              '- 重建基线（全量 2400 + ⑥校准）综合 0.5688', '',
              '所有数字为本地官方口径复刻，非官方排行榜成绩。val 反复比较属探索性结论。']
    (N / 'FINAL_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    state(status='overnight_complete', stage='final_report')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        state(status='failed', stage='overnight', error=str(exc))
        traceback.print_exc()
        raise
