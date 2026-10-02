#!/usr/bin/env python3
"""
8 小时无人值守流水线 (幂等 + 可续跑):
  1. train_r16   : Qwen3-14B LoRA r16 2ep   -> outputs/qwen3-14b-lora   (断点续训)
  2. eval_base   : 基座零训练 few-shot 打 val -> eval/report_base.json
  3. eval_r16    : SFT r16 打 val             -> eval/report_r16.json
  4. train_r32   : Qwen3-14B LoRA r32 2ep     -> outputs/qwen3-14b-lora-r32 (断点续训)
  5. eval_r32    : SFT r32 打 val             -> eval/report_r32.json
  6. predict_test: 选 val 最优的模型生成 test  -> outputs/result.jsonl (+ array)
每个阶段: 有完成标记则跳过; 失败自动重试; 每次开始前清理残留进程并等 GPU 空闲。
"""
import os, sys, subprocess, time, json, glob

ROOT = "/data/neg_opinion"
PY = "/home/yuanhuilin/miniconda3/envs/YHLin/bin/python"
OUT = f"{ROOT}/outputs"
EVAL = f"{ROOT}/eval"
DATA = f"{ROOT}/data"
os.makedirs(OUT, exist_ok=True)
os.makedirs(EVAL, exist_ok=True)

def log(*a):
    msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(f"{OUT}/pipeline.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def set_status(s):
    with open(f"{OUT}/STATUS", "w", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}\n")

def sh(cmd, logfile):
    log("RUN:", " ".join(cmd))
    with open(logfile, "a", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        return p.wait()

def gpu_used_mb():
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL).decode()
        return max(int(x) for x in o.split())
    except Exception:
        return 10 ** 9

def kill_stale_train():
    subprocess.run("pkill -f 'src/train_lora.py'", shell=True)
    subprocess.run("pkill -f 'src/infer.py'", shell=True)
    time.sleep(6)

def wait_gpu_free(limit_mb=3500, timeout=1200):
    t = time.time()
    while gpu_used_mb() > limit_mb:
        if time.time() - t > timeout:
            log("GPU busy too long, killing stale procs"); kill_stale_train(); t = time.time()
        log("wait GPU free, used MB =", gpu_used_mb()); time.sleep(30)

# ----------------- 阶段实现 -----------------
def stage_train(tag, adapter_dir, extra):
    if os.path.exists(f"{adapter_dir}/.DONE") and os.path.exists(f"{adapter_dir}/adapter_config.json"):
        log(f"skip {tag} (done)"); return True
    set_status(f"train {tag}")
    kill_stale_train(); wait_gpu_free()
    cmd = [PY, "-u", "src/train_lora.py", "--out", adapter_dir] + extra
    rc = sh(cmd, f"{OUT}/{tag}.train.log")
    if rc == 0 and os.path.exists(f"{adapter_dir}/adapter_config.json"):
        open(f"{adapter_dir}/.DONE", "w").write("ok"); log(f"{tag} DONE"); return True
    log(f"{tag} FAILED rc={rc}"); return False

def stage_infer(tag, adapter, data, pred_out, extra=None):
    marker = f"{pred_out}.DONE"
    if os.path.exists(marker):
        log(f"skip {tag} (done)"); return True
    set_status(f"infer {tag}")
    wait_gpu_free()
    cmd = [PY, "-u", "src/infer.py", "--data", data, "--out", pred_out] + (extra or [])
    if adapter:
        cmd += ["--adapter", adapter]
    rc = sh(cmd, f"{OUT}/{tag}.infer.log")
    # 需要条数完整
    try:
        n = sum(1 for _ in open(pred_out, encoding="utf-8"))
    except Exception:
        n = 0
    want = sum(1 for _ in open(data, encoding="utf-8"))
    if rc == 0 and n >= want:
        open(marker, "w").write("ok"); log(f"{tag} DONE"); return True
    log(f"{tag} incomplete rc={rc} {n}/{want}"); return False

def stage_score(tag, pred, gold, report):
    if os.path.exists(report):
        log(f"skip {tag} (done)"); return True
    set_status(f"score {tag}")
    rc = sh([PY, "-u", "src/eval_scorer.py", "--gold", gold, "--pred", pred,
             "--device", "cuda", "--out", report], f"{OUT}/{tag}.score.log")
    if rc == 0 and os.path.exists(report):
        log(f"{tag} DONE"); return True
    log(f"{tag} FAILED rc={rc}"); return False

def composite(report_path):
    try:
        return json.load(open(report_path, encoding="utf-8"))["report"]["composite"]
    except Exception:
        return -1.0

def stage_predict_test():
    if os.path.exists(f"{OUT}/result.jsonl.DONE"):
        log("skip predict_test (done)"); return True
    set_status("predict test")
    # 选 val 最优
    cands = [(composite(f"{EVAL}/report_base.json"), None),
             (composite(f"{EVAL}/report_r16.json"), f"{OUT}/qwen3-14b-lora"),
             (composite(f"{EVAL}/report_r32.json"), f"{OUT}/qwen3-14b-lora-r32")]
    cands = [c for c in cands if c[0] >= 0]
    if not cands:
        log("no reports, fallback to base"); best = None
    else:
        best = max(cands, key=lambda x: x[0])
    log("best for test:", "base" if best[1] is None else best[1], "composite=", best[0])
    ok = stage_infer("infer_test", best[1], f"{DATA}/sft_test.jsonl", f"{OUT}/result.jsonl")
    if not ok:
        return False
    # 同时输出数组版本 (赛题格式有歧义)
    rows = [json.loads(l) for l in open(f"{OUT}/result.jsonl", encoding="utf-8")]
    json.dump(rows, open(f"{OUT}/result_array.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(f"{OUT}/result.jsonl.DONE", "w").write("ok")
    return True

# ----------------- 主流程 -----------------
def run_stage(fn, *args, retries=4):
    for i in range(retries):
        try:
            if fn(*args):
                return True
        except Exception as e:
            log(f"stage error {fn.__name__}: {type(e).__name__}: {e}")
        log(f"retry {fn.__name__} {i+1}/{retries} in 60s")
        time.sleep(60)
    return False

def main():
    log("=" * 20, "PIPELINE START", "=" * 20)
    run_stage(stage_train, "train_r16", f"{OUT}/qwen3-14b-lora",
              ["--epochs", "2", "--max-len", "6144", "--lora-r", "16", "--lora-alpha", "32",
               "--lr", "1e-4", "--grad-accum", "16", "--save-steps", "50", "--logging-steps", "5"])
    run_stage(stage_infer, "infer_base_val", None, f"{DATA}/sft_val.jsonl", f"{EVAL}/pred_base_val.jsonl")
    run_stage(stage_score, "score_base", f"{EVAL}/pred_base_val.jsonl", f"{DATA}/val.jsonl", f"{EVAL}/report_base.json")
    run_stage(stage_infer, "infer_r16_val", f"{OUT}/qwen3-14b-lora", f"{DATA}/sft_val.jsonl", f"{EVAL}/pred_r16_val.jsonl")
    run_stage(stage_score, "score_r16", f"{EVAL}/pred_r16_val.jsonl", f"{DATA}/val.jsonl", f"{EVAL}/report_r16.json")
    run_stage(stage_train, "train_r32", f"{OUT}/qwen3-14b-lora-r32",
              ["--epochs", "2", "--max-len", "6144", "--lora-r", "32", "--lora-alpha", "64",
               "--lr", "1e-4", "--grad-accum", "16", "--save-steps", "50", "--logging-steps", "5"])
    run_stage(stage_infer, "infer_r32_val", f"{OUT}/qwen3-14b-lora-r32", f"{DATA}/sft_val.jsonl", f"{EVAL}/pred_r32_val.jsonl")
    run_stage(stage_score, "score_r32", f"{EVAL}/pred_r32_val.jsonl", f"{DATA}/val.jsonl", f"{EVAL}/report_r32.json")
    run_stage(stage_predict_test)

    # 汇总
    summary = {"base": composite(f"{EVAL}/report_base.json"),
               "lora_r16": composite(f"{EVAL}/report_r16.json"),
               "lora_r32": composite(f"{EVAL}/report_r32.json")}
    json.dump(summary, open(f"{OUT}/SUMMARY.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    open(f"{OUT}/PIPELINE_DONE", "w").write("ok")
    set_status("PIPELINE DONE")
    log("SUMMARY:", json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
