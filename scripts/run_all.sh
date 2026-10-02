#!/bin/bash
# 无人值守 wrapper: pipeline 退出后自动重启; 全部完成则退出。
cd /data/neg_opinion || exit 1
PY=/home/yuanhuilin/miniconda3/envs/YHLin/bin/python
mkdir -p outputs eval
echo "[run_all] started $(date)" >> outputs/pipeline.log
while true; do
  "$PY" -u src/pipeline.py >> outputs/pipeline.log 2>&1
  if [ -f outputs/PIPELINE_DONE ]; then
    echo "[run_all] PIPELINE_DONE, exiting $(date)" >> outputs/pipeline.log
    break
  fi
  echo "[run_all] pipeline exited rc=$?, restart in 30s $(date)" >> outputs/pipeline.log
  sleep 30
done
