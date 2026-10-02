#!/bin/bash
cd /data/neg_opinion || exit 1
mkdir -p outputs
while true; do
  if [ -f outputs/POST_DONE ]; then
    echo "[watchdog_post] POST_DONE, exit $(date)" >> outputs/watchdog_post.log
    break
  fi
  if ! pgrep -f 'bash src/run_post.sh' >/dev/null 2>&1; then
    echo "[watchdog_post] restart run_post $(date)" >> outputs/watchdog_post.log
    ( setsid bash src/run_post.sh >/dev/null 2>&1 < /dev/null & )
  fi
  sleep 180
done
