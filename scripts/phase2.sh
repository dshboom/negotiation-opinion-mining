#!/bin/bash
# Phase 2: SFT v2 —— 字段加权 (v2w) vs 无加权对照 (v2c), 各 3 epochs
#   每个模型走同一验收链: vLLM T=0 推理 -> rescore -> 按同配方重拟⑥ -> 全量评分
#   断点续跑: 每步产物存在即跳过; 训练崩了从 checkpoint 恢复
cd /data/neg_opinion
YH=/home/yuanhuilin/miniconda3/envs/YHLin/bin/python
AC=/data/neg_opinion/.venv-accel/bin/python
LOG=outputs/phase2.log
mkdir -p outputs/diag eval
st(){ echo "$(date '+%F %T') $1" | tee -a outputs/STATUS_P2; echo "$1" > outputs/STATUS_P2.cur; }
clean_gpu(){
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$USED" ] && [ "$USED" -gt 10000 ]; then
    st "gpu_busy_${USED}MiB cleanup"
    for pid in $(ps -eo pid,args | grep -E 'vllm|EngineCore' | grep -v grep | awk '{print $1}'); do
      kill -9 $pid 2>/dev/null
    done
    sleep 8
  fi
}

# 等 Phase 1 释放 GPU
for i in $(seq 1 60); do [ -f outputs/P1_DONE ] && break; sleep 30; done
clean_gpu

# ---- 0) v2 加权定位自检 ----
if [ ! -f outputs/diag/v2_selftest.ok ]; then
  st "v2 selftest"
  $YH -u src/train_lora_v2.py --selftest --out outputs/qwen3-14b-lora-r32-v2w > outputs/diag/v2_selftest.txt 2>&1
  if grep -q "Traceback" outputs/diag/v2_selftest.txt; then
    st "v2 selftest FAILED (traceback)"; cat outputs/diag/v2_selftest.txt; exit 1
  fi
  if ! grep -q "fallback率: 0 " outputs/diag/v2_selftest.txt; then
    st "v2 selftest FAILED (fallback>0)"; cat outputs/diag/v2_selftest.txt; exit 1
  fi
  touch outputs/diag/v2_selftest.ok
  cat outputs/diag/v2_selftest.txt
fi

# ---- 1) 训练 v2w (字段加权: name×2 stance×3 fut×2, 3ep) ----
if [ ! -f outputs/qwen3-14b-lora-r32-v2w/COMPLETED ]; then
  clean_gpu
  st "train v2w (w:name2 stance3 fut2, 3ep, ~3h)"
  $YH -u src/train_lora_v2.py --out outputs/qwen3-14b-lora-r32-v2w \
      --epochs 3 --lora-r 32 --lora-alpha 64 --w-name 2 --w-stance 3 --w-fut 2 >> $LOG 2>&1 \
      && touch outputs/qwen3-14b-lora-r32-v2w/COMPLETED
fi

# ---- 2) v2w 验收链 ----
if [ -f outputs/qwen3-14b-lora-r32-v2w/COMPLETED ] && [ ! -f outputs/diag/v2w.cal6.json ]; then
  clean_gpu
  st "infer v2w (T=0)"
  $AC -u src/infer_vllm.py --data data/sft_val.jsonl --out eval/pred_v2w_val.jsonl \
      --adapter outputs/qwen3-14b-lora-r32-v2w >> $LOG 2>&1
  st "rescore v2w"
  $YH -u src/rescore.py --data data/sft_val.jsonl --pred eval/pred_v2w_val.jsonl \
      --out eval/rescore_v2w_val.jsonl --adapter outputs/qwen3-14b-lora-r32-v2w >> $LOG 2>&1
  $YH -u src/fit_bias.py --rescore eval/rescore_v2w_val.jsonl --out outputs/diag/v2w.bias.json >> $LOG 2>&1
  BN=$($YH -c "import json;print(json.load(open('outputs/diag/v2w.bias.json'))['b_neutral'])")
  BO=$($YH -c "import json;print(json.load(open('outputs/diag/v2w.bias.json'))['b_oppose'])")
  st "apply6+eval v2w (bias=$BN,$BO)"
  $YH -u src/apply_stance_calib.py --pred eval/pred_v2w_val.jsonl --rescore eval/rescore_v2w_val.jsonl \
      --b-neutral $BN --b-oppose $BO --out outputs/diag/v2w.cal6.jsonl >> $LOG 2>&1
  $YH -u src/eval_scorer.py --gold data/val.jsonl --pred outputs/diag/v2w.cal6.jsonl \
      --device cuda --out outputs/diag/v2w.cal6.json >> $LOG 2>&1
fi

# ---- 3) 训练 v2c (对照: 无加权, 3ep) ----
if [ ! -f outputs/qwen3-14b-lora-r32-v2c/COMPLETED ]; then
  clean_gpu
  st "train v2c (control, 3ep, ~3h)"
  $YH -u src/train_lora.py --out outputs/qwen3-14b-lora-r32-v2c \
      --epochs 3 --lora-r 32 --lora-alpha 64 >> $LOG 2>&1 \
      && touch outputs/qwen3-14b-lora-r32-v2c/COMPLETED
fi

# ---- 4) v2c 验收链 ----
if [ -f outputs/qwen3-14b-lora-r32-v2c/COMPLETED ] && [ ! -f outputs/diag/v2c.cal6.json ]; then
  clean_gpu
  st "infer v2c (T=0)"
  $AC -u src/infer_vllm.py --data data/sft_val.jsonl --out eval/pred_v2c_val.jsonl \
      --adapter outputs/qwen3-14b-lora-r32-v2c >> $LOG 2>&1
  st "rescore v2c"
  $YH -u src/rescore.py --data data/sft_val.jsonl --pred eval/pred_v2c_val.jsonl \
      --out eval/rescore_v2c_val.jsonl --adapter outputs/qwen3-14b-lora-r32-v2c >> $LOG 2>&1
  $YH -u src/fit_bias.py --rescore eval/rescore_v2c_val.jsonl --out outputs/diag/v2c.bias.json >> $LOG 2>&1
  BN=$($YH -c "import json;print(json.load(open('outputs/diag/v2c.bias.json'))['b_neutral'])")
  BO=$($YH -c "import json;print(json.load(open('outputs/diag/v2c.bias.json'))['b_oppose'])")
  st "apply6+eval v2c (bias=$BN,$BO)"
  $YH -u src/apply_stance_calib.py --pred eval/pred_v2c_val.jsonl --rescore eval/rescore_v2c_val.jsonl \
      --b-neutral $BN --b-oppose $BO --out outputs/diag/v2c.cal6.jsonl >> $LOG 2>&1
  $YH -u src/eval_scorer.py --gold data/val.jsonl --pred outputs/diag/v2c.cal6.jsonl \
      --device cuda --out outputs/diag/v2c.cal6.json >> $LOG 2>&1
fi

st "P2 DONE"
date > outputs/P2_DONE
echo "PHASE2 ALL DONE"
