#!/usr/bin/env python3
"""
后处理流水线 (在主流水线完成后运行; 幂等 + 可续跑):
 P0 等主流水线完成
 P1 前向打分 val (r32)                      -> eval/rescore_r32_val.jsonl
 P2 生成 calib(train 子集 240) + 前向打分     -> eval/pred_calib.jsonl, rescore_calib.jsonl   [⑥b 用]
 P3 postprocess: ⑥a/⑥b 立场校准, ⑦ k扫描, ⑧ 去重
 P4 ⑦ 探针: 用"最多7张卡"指令生成 20 条, 看能否>5张
 P5 若可行: 全量生成 val(k_max) + 打分 -> postprocess tag=k7  (⑦ 扫到 k=7)
"""
import os, sys, json, subprocess, time

ROOT = "/data/neg_opinion"
PY = "/home/yuanhuilin/miniconda3/envs/YHLin/bin/python"
ACC = "/data/neg_opinion/.venv-accel/bin/python"   # vLLM 环境
OUT = f"{ROOT}/outputs"
EVAL = f"{ROOT}/eval"
DATA = f"{ROOT}/data"
POST = f"{OUT}/post"
K7_EXTRA = "如果原文中还存在其他议题，请尽量输出 6~7 个议题（含主议题），不要遗漏证据充分的议题。"
os.makedirs(POST, exist_ok=True)

def log(*a):
    m = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " + " ".join(str(x) for x in a)
    print(m, flush=True)
    open(f"{OUT}/postpipeline.log", "a", encoding="utf-8").write(m + "\n")

def status(s):
    open(f"{OUT}/STATUS_POST", "w", encoding="utf-8").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}\n")

def sh(cmd, logfile):
    log("RUN:", " ".join(cmd))
    with open(logfile, "a", encoding="utf-8") as lf:
        return subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True).wait()

def gpu_mb():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                     "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode()
        return max(int(x) for x in o.split())
    except Exception:
        return 10 ** 9

def kill_stale():
    subprocess.run("pkill -f 'src/infer.py'", shell=True)
    subprocess.run("pkill -f 'src/infer_vllm.py'", shell=True)
    subprocess.run("pkill -f 'src/rescore.py'", shell=True)
    time.sleep(6)

def wait_gpu_free(limit=3500, timeout=1800):
    t = time.time()
    while gpu_mb() > limit:
        if time.time() - t > timeout:
            kill_stale(); t = time.time()
        log("wait GPU, used MB=", gpu_mb()); time.sleep(30)

def run(fn, *a, retries=4):
    for i in range(retries):
        try:
            if fn(*a): return True
        except Exception as e:
            log(f"ERR {fn.__name__}: {type(e).__name__}: {e}")
        log(f"retry {fn.__name__} {i+1}/{retries}"); time.sleep(60)
    return False

# ---------- stages ----------
def wait_main():
    if os.path.exists(f"{OUT}/PIPELINE_DONE"):
        return True
    log("waiting for main PIPELINE_DONE ...")
    status("wait main pipeline")
    t = time.time()
    while not os.path.exists(f"{OUT}/PIPELINE_DONE"):
        if time.time() - t > 12 * 3600:
            return False
        time.sleep(60)
    return True

def rescore(tag, adapter, data, pred, out):
    if os.path.exists(out + ".DONE"): return True
    status(f"rescore {tag}"); wait_gpu_free()
    cmd = [PY, "-u", "src/rescore.py", "--data", data, "--pred", pred, "--out", out] + (["--adapter", adapter] if adapter else [])
    rc = sh(cmd, f"{OUT}/{tag}.rescore.log")
    want = sum(1 for _ in open(pred, encoding="utf-8"))
    n = sum(1 for _ in open(out, encoding="utf-8")) if os.path.exists(out) else 0
    if rc == 0 and n >= want:
        open(out + ".DONE", "w").write("ok"); return True
    log(f"rescore {tag} incomplete {n}/{want}"); return False

def infer(tag, adapter, data, out, extra=None, limit=0):
    marker = out + ".DONE"
    if os.path.exists(marker): return True
    status(f"infer {tag}"); wait_gpu_free()
    cmd = [ACC, "-u", "src/infer_vllm.py", "--data", data, "--out", out]
    if adapter: cmd += ["--adapter", adapter]
    if extra: cmd += ["--extra", extra]
    if limit: cmd += ["--limit", str(limit)]
    rc = sh(cmd, f"{OUT}/{tag}.infer.log")
    want = min(limit, sum(1 for _ in open(data, encoding="utf-8"))) if limit else sum(1 for _ in open(data, encoding="utf-8"))
    n = sum(1 for _ in open(out, encoding="utf-8")) if os.path.exists(out) else 0
    if rc == 0 and n >= want:
        open(marker, "w").write("ok"); return True
    log(f"infer {tag} incomplete {n}/{want}"); return False

def stage_test_vllm():
    out = f"{OUT}/result_vllm.jsonl"
    if os.path.exists(out + ".DONE"): return True
    ok = infer("infer_test_vllm", f"{OUT}/qwen3-14b-lora-r32", f"{DATA}/sft_test.jsonl", out)
    if not ok: return False
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    json.dump(rows, open(f"{OUT}/result_vllm_array.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return True

def postproc(tag, pred, rescore_file, calib=False):
    outp = f"{POST}/postprocess_{tag}.json"
    if os.path.exists(outp): return True
    status(f"postprocess {tag}"); wait_gpu_free(limit=5000)
    cmd = [PY, "-u", "src/postprocess.py", "--pred", pred, "--rescore", rescore_file,
           "--tag", tag, "--outdir", POST]
    if not calib:
        cmd += ["--calib-pred", "__none__"]
    rc = sh(cmd, f"{OUT}/postprocess_{tag}.log")
    return rc == 0 and os.path.exists(outp)

def make_calib(n=240):
    sft = f"{DATA}/sft_calib.jsonl"; gld = f"{DATA}/calib_gold.jsonl"
    if not (os.path.exists(sft) and os.path.exists(gld)):
        log("make calib slices", n)
        open(sft, "w").write("".join(open(f"{DATA}/sft_train.jsonl", encoding="utf-8").readlines()[:n]))
        open(gld, "w").write("".join(open(f"{DATA}/train.jsonl", encoding="utf-8").readlines()[:n]))
    return sft, gld

def main():
    log("=" * 15, "POST PIPELINE START", "=" * 15)
    run(wait_main)

    # P1: rescore val (r32)
    run(rescore, "rescore_r32_val", f"{OUT}/qwen3-14b-lora-r32",
        f"{DATA}/sft_val.jsonl", f"{EVAL}/pred_r32_val.jsonl", f"{EVAL}/rescore_r32_val.jsonl")

    # P2: calib gen + rescore (⑥b)
    make_calib(240)
    run(infer, "infer_calib", f"{OUT}/qwen3-14b-lora-r32", f"{DATA}/sft_calib.jsonl", f"{EVAL}/pred_calib.jsonl")
    run(rescore, "rescore_calib", f"{OUT}/qwen3-14b-lora-r32", f"{DATA}/sft_calib.jsonl",
        f"{EVAL}/pred_calib.jsonl", f"{EVAL}/rescore_calib.jsonl")

    # P3: postprocess (⑥a/⑥b/⑦/⑧)
    run(postproc, "r32", f"{EVAL}/pred_r32_val.jsonl", f"{EVAL}/rescore_r32_val.jsonl", True)

    # P4: k1 probe for "<=7 cards"
    run(infer, "probe_k7", f"{OUT}/qwen3-14b-lora-r32", f"{DATA}/sft_val.jsonl",
        f"{EVAL}/probe_k7.jsonl", K7_EXTRA, 20)
    mean_cards = 0
    try:
        rows = [json.loads(l) for l in open(f"{EVAL}/probe_k7.jsonl", encoding="utf-8") if l.strip()]
        mean_cards = sum(len(r.get("issue_list") or []) for r in rows) / max(len(rows), 1)
    except Exception:
        pass
    log(f"probe mean cards = {mean_cards:.2f}")

    if mean_cards > 5.2:
        run(infer, "infer_k7_val", f"{OUT}/qwen3-14b-lora-r32", f"{DATA}/sft_val.jsonl",
            f"{EVAL}/pred_k7_val.jsonl", K7_EXTRA)
        run(rescore, "rescore_k7_val", f"{OUT}/qwen3-14b-lora-r32", f"{DATA}/sft_val.jsonl",
            f"{EVAL}/pred_k7_val.jsonl", f"{EVAL}/rescore_k7_val.jsonl")
        run(postproc, "k7", f"{EVAL}/pred_k7_val.jsonl", f"{EVAL}/rescore_k7_val.jsonl", False)
    else:
        log("probe shows model stays <=5 cards; skip k7 full run")

    # P6: 用 vLLM 快速生成最终 test 预测
    run(stage_test_vllm)

    open(f"{OUT}/POST_DONE", "w").write("ok")
    status("POST DONE")
    log("POST PIPELINE DONE")

if __name__ == "__main__":
    main()
