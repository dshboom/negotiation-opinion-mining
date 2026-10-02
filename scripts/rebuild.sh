#!/bin/bash
# 重建 Stage 2: 训练 r32 -> val评测(⑥) -> test预测 -> 最终提交 (断点续跑)
cd /data/neg_opinion || exit 1
YH=/home/yuanhuilin/miniconda3/envs/YHLin/bin/python
AC=/data/neg_opinion/.venv-accel/bin/python
LOG=outputs/rebuild.log
mkdir -p outputs/diag eval
st(){ echo "$(date '+%F %T') $1" | tee -a outputs/STATUS_REBUILD; echo "$1" > outputs/STATUS_REBUILD.cur; }
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

# 0) 等 Stage 1
for i in $(seq 1 360); do [ -f outputs/SETUP_DONE ] && break; sleep 30; done
clean_gpu

# 0.5) 训练冒烟 (2 步, 验证环境/脚本兼容)
if [ ! -f outputs/diag/train_smoke.ok ]; then
  st "train smoke (2 steps)"
  $YH -u src/train_lora.py --out outputs/smoke-adapter --epochs 1 --max-steps 2 \
      --lora-r 8 --lora-alpha 16 --save-steps 1000 >> $LOG 2>&1 && touch outputs/diag/train_smoke.ok
fi

# 1) 训练 r32 2ep
if [ ! -f outputs/qwen3-14b-lora-r32/adapter_config.json ]; then
  clean_gpu; st "train r32 2ep (~2h)"
  $YH -u src/train_lora.py --out outputs/qwen3-14b-lora-r32 --epochs 2 --max-len 6144 \
      --lora-r 32 --lora-alpha 64 --lr 1e-4 --grad-accum 16 --save-steps 50 --logging-steps 5 >> $LOG 2>&1
fi
st "r32 adapter: $(ls outputs/qwen3-14b-lora-r32/adapter_config.json 2>/dev/null || echo MISSING)"

# 2) val 推理 (vLLM T=0)
if [ ! -f eval/pred_r32_val.jsonl ]; then
  clean_gpu; st "infer r32 val (vllm T=0)"
  $AC -u src/infer_vllm.py --data data/sft_val.jsonl --out eval/pred_r32_val.jsonl \
      --adapter outputs/qwen3-14b-lora-r32 >> $LOG 2>&1
fi

# 3) rescore + 拟合 ⑥ 偏置
if [ ! -f eval/rescore_r32_val.jsonl ]; then
  st "rescore r32 val"
  $YH -u src/rescore.py --data data/sft_val.jsonl --pred eval/pred_r32_val.jsonl \
      --out eval/rescore_r32_val.jsonl --adapter outputs/qwen3-14b-lora-r32 >> $LOG 2>&1
fi
if [ ! -f outputs/diag/r32.bias.json ]; then
  $YH -u src/fit_bias.py --rescore eval/rescore_r32_val.jsonl --out outputs/diag/r32.bias.json >> $LOG 2>&1
fi
BN=$($YH -c "import json;print(json.load(open('outputs/diag/r32.bias.json'))['b_neutral'])" 2>/dev/null)
BO=$($YH -c "import json;print(json.load(open('outputs/diag/r32.bias.json'))['b_oppose'])" 2>/dev/null)

# 4) 评分 raw + ⑥
if [ ! -f outputs/diag/r32.raw.json ]; then
  st "eval r32 raw"
  $YH -u src/eval_scorer.py --gold data/val.jsonl --pred eval/pred_r32_val.jsonl \
      --device cuda --out outputs/diag/r32.raw.json >> $LOG 2>&1
fi
if [ ! -f outputs/diag/r32.cal6.json ]; then
  st "apply6 + eval (bias=$BN,$BO)"
  $YH -u src/apply_stance_calib.py --pred eval/pred_r32_val.jsonl --rescore eval/rescore_r32_val.jsonl \
      --b-neutral $BN --b-oppose $BO --out outputs/diag/r32.cal6.jsonl >> $LOG 2>&1
  $YH -u src/eval_scorer.py --gold data/val.jsonl --pred outputs/diag/r32.cal6.jsonl \
      --device cuda --out outputs/diag/r32.cal6.json >> $LOG 2>&1
fi

# 5) test 预测 + ⑥ -> 最终提交
if [ ! -f outputs/result_vllm.jsonl ]; then
  clean_gpu; st "infer test (vllm)"
  $AC -u src/infer_vllm.py --data data/sft_test.jsonl --out outputs/result_vllm.jsonl \
      --adapter outputs/qwen3-14b-lora-r32 >> $LOG 2>&1
fi
if [ ! -f eval/rescore_r32_test.jsonl ]; then
  st "rescore test"
  $YH -u src/rescore.py --data data/sft_test.jsonl --pred outputs/result_vllm.jsonl \
      --out eval/rescore_r32_test.jsonl --adapter outputs/qwen3-14b-lora-r32 >> $LOG 2>&1
fi
if [ ! -f outputs/result_final.jsonl ]; then
  st "final submission"
  $YH -u src/apply_stance_calib.py --pred outputs/result_vllm.jsonl --rescore eval/rescore_r32_test.jsonl \
      --b-neutral $BN --b-oppose $BO --out outputs/result_final.jsonl \
      --out-array outputs/result_final_array.json >> $LOG 2>&1
fi
st "REBUILD DONE"
date > outputs/REBUILD_DONE
