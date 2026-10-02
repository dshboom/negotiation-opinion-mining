#!/bin/bash
cd /data/neg_opinion || exit 1
PY=/home/yuanhuilin/miniconda3/envs/YHLin/bin/python
mkdir -p outputs/post
echo "[run_post] started $(date)" >> outputs/postpipeline.log
while true; do
  "$PY" -u src/postpipeline.py >> outputs/postpipeline.log 2>&1
  if [ -f outputs/POST_DONE ]; then
    echo "[run_post] POST_DONE, exit $(date)" >> outputs/postpipeline.log
    break
  fi
  echo "[run_post] exited rc=$?, restart in 30s $(date)" >> outputs/postpipeline.log
  sleep 30
done
