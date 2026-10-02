#!/bin/bash
# Phase 1: 指令探针B1/B2 -> n=5采样 -> SC-merge(F/thr变体+single-best) -> rescore+⑥ -> 全量评分
# 断点续跑: 每步产物存在即跳过
cd /data/neg_opinion
YH=/home/yuanhuilin/miniconda3/envs/YHLin/bin/python
AC=/data/neg_opinion/.venv-accel/bin/python
mkdir -p outputs/diag eval
LOG=outputs/phase1.log
st(){ echo "$(date '+%F %T') $1" | tee -a outputs/STATUS_P1; echo "$1" > outputs/STATUS_P1.cur; }

# GPU 残留清理 (vLLM EngineCore 僵尸)
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ -n "$USED" ] && [ "$USED" -gt 10000 ]; then
  st "gpu_busy_${USED}MiB cleanup"
  for pid in $(ps -eo pid,args | grep -E 'vllm|EngineCore' | grep -v grep | awk '{print $1}'); do
    kill -9 $pid 2>/dev/null
  done
  sleep 8
fi

# ---- Step 1: 探针 B1 (命名口径) ----
if [ ! -f outputs/diag/probeB1.json ]; then
  st "probeB1 infer"
  $AC -u src/infer_vllm.py --data data/sft_val.jsonl --out eval/probeB1_val.jsonl \
      --extra "注意：issue_name 请用简明概括的主题词命名（不要照抄原文整句）；argument_chain 必须逐字引用原文原句。" >> $LOG 2>&1
  st "probeB1 eval"
  $YH -u src/eval_scorer.py --gold data/val.jsonl --pred eval/probeB1_val.jsonl \
      --device cuda --out outputs/diag/probeB1.json >> $LOG 2>&1
fi

# ---- Step 2: 探针 B2 (切分粒度) ----
if [ ! -f outputs/diag/probeB2.json ]; then
  st "probeB2 infer"
  $AC -u src/infer_vllm.py --data data/sft_val.jsonl --out eval/probeB2_val.jsonl \
      --extra "注意：请为每个独立议题单独建卡，一个议题一张卡，不要把多个议题合并进同一张卡。" >> $LOG 2>&1
  st "probeB2 eval"
  $YH -u src/eval_scorer.py --gold data/val.jsonl --pred eval/probeB2_val.jsonl \
      --device cuda --out outputs/diag/probeB2.json >> $LOG 2>&1
fi

# ---- Step 3: n=5 采样 (T=0.7, top_p=0.8) ----
if [ ! -f eval/samples_r32_val.jsonl.run4 ]; then
  st "sampling n=5"
  $AC -u src/sample_vllm.py --data data/sft_val.jsonl --out eval/samples_r32_val.jsonl \
      --n 5 --temperature 0.7 --top-p 0.8 >> $LOG 2>&1
fi

# ---- Step 4: SC-merge 变体 -> rescore -> ⑥ -> 评分 ----
for cfg in "2 0.80 F2t80" "3 0.80 F3t80" "2 0.65 F2t65"; do
  set -- $cfg; F=$1; THR=$2; TAG=$3
  if [ ! -f outputs/diag/merged_$TAG.cal6.json ]; then
    st "merge $TAG"
    $YH -u src/sc_merge.py --prefix eval/samples_r32_val.jsonl --n 5 --freq $F --thr $THR \
        --out outputs/diag/merged_$TAG.jsonl >> $LOG 2>&1
    $YH -u src/rescore.py --data data/sft_val.jsonl --pred outputs/diag/merged_$TAG.jsonl \
        --out eval/rescore_merged_$TAG.jsonl --adapter outputs/qwen3-14b-lora-r32 >> $LOG 2>&1
    $YH -u src/apply_stance_calib.py --pred outputs/diag/merged_$TAG.jsonl \
        --rescore eval/rescore_merged_$TAG.jsonl --b-neutral 0.7 --b-oppose -0.5 \
        --out outputs/diag/merged_$TAG.cal6.jsonl >> $LOG 2>&1
    st "eval merged_$TAG"
    $YH -u src/eval_scorer.py --gold data/val.jsonl --pred outputs/diag/merged_$TAG.cal6.jsonl \
        --device cuda --out outputs/diag/merged_$TAG.cal6.json >> $LOG 2>&1
  fi
done

# ---- Step 5: single-best 消融 ----
if [ ! -f outputs/diag/singlebest.cal6.json ]; then
  st "merge single-best"
  $YH -u src/sc_merge.py --prefix eval/samples_r32_val.jsonl --n 5 --single-best \
      --out outputs/diag/singlebest.jsonl >> $LOG 2>&1
  $YH -u src/rescore.py --data data/sft_val.jsonl --pred outputs/diag/singlebest.jsonl \
      --out eval/rescore_singlebest.jsonl --adapter outputs/qwen3-14b-lora-r32 >> $LOG 2>&1
  $YH -u src/apply_stance_calib.py --pred outputs/diag/singlebest.jsonl \
      --rescore eval/rescore_singlebest.jsonl --b-neutral 0.7 --b-oppose -0.5 \
      --out outputs/diag/singlebest.cal6.jsonl >> $LOG 2>&1
  st "eval singlebest"
  $YH -u src/eval_scorer.py --gold data/val.jsonl --pred outputs/diag/singlebest.cal6.jsonl \
      --device cuda --out outputs/diag/singlebest.cal6.json >> $LOG 2>&1
fi

st "P1 DONE"
date > outputs/P1_DONE
echo "PHASE1 ALL DONE"
