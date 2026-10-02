#!/bin/bash
# 第二层守护: 若 run_all 不在运行且流水线未完成, 就把它拉起来。独立于 run_all。
cd /data/neg_opinion || exit 1
mkdir -p outputs
while true; do
  if [ -f outputs/PIPELINE_DONE ]; then
    echo "[watchdog] PIPELINE_DONE, exit $(date)" >> outputs/watchdog.log
    break
  fi
  if ! pgrep -f 'bash src/run_all.sh' >/dev/null 2>&1; then
    echo "[watchdog] run_all not running, restart $(date)" >> outputs/watchdog.log
    ( setsid bash src/run_all.sh >/dev/null 2>&1 < /dev/null & )
  fi
  sleep 180
done
