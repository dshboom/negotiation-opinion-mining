#!/bin/bash
# 重建 Stage 1: 目录/数据/环境/模型下载/冒烟 (断点续跑)
cd /data/neg_opinion || exit 1
VENV=/data/neg_opinion/.venv-hf
ACC=/data/neg_opinion/.venv-accel
YH=/home/yuanhuilin/miniconda3/envs/YHLin/bin/python
M="-i https://pypi.tuna.tsinghua.edu.cn/simple"
LOG=outputs/setup.log
mkdir -p src data eval outputs models logs diag
st(){ echo "$(date '+%F %T') $1" | tee -a outputs/STATUS_SETUP; }

# --- 0. 数据构建 (system python3, 纯标准库) ---
st "build sft data"
python3 src/build_sft.py --inp data/train.jsonl --out data/sft_train.jsonl --labeled >> $LOG 2>&1
python3 src/build_sft.py --inp data/val.jsonl   --out data/sft_val.jsonl   --labeled >> $LOG 2>&1
python3 src/build_sft.py --inp data/test.jsonl  --out data/sft_test.jsonl >> $LOG 2>&1
head -400 data/sft_train.jsonl > data/sft_calib.jsonl
head -400 data/train.jsonl     > data/calib_gold.jsonl
st "data built: $(wc -l < data/sft_train.jsonl)/$(wc -l < data/sft_val.jsonl)/$(wc -l < data/sft_test.jsonl)"

# --- 1. venvs + 软链旧路径 ---
st "create venvs"
[ -d $VENV ] || python3 -m virtualenv -q $VENV
[ -d $ACC  ] || python3 -m virtualenv -q $ACC
mkdir -p /home/yuanhuilin/miniconda3/envs
ln -sfn $VENV /home/yuanhuilin/miniconda3/envs/YHLin
$VENV/bin/pip install -q -U pip $M >> $LOG 2>&1

# --- 2. requests + 后台启动模型下载 ---
$VENV/bin/pip install -q $M requests >> $LOG 2>&1
if [ ! -f outputs/.dl_done ]; then
  st "start downloads (bge + bert + Qwen3-14B 27.5GiB)"
  setsid nohup bash -c '
    cd /data/neg_opinion
    P=/data/neg_opinion/.venv-hf/bin/python
    $P src/ms_dl.py AI-ModelScope/bge-small-zh-v1.5 models/bge-small-zh-v1.5
    $P src/ms_dl.py tiansz/bert-base-chinese models/bert-base-chinese
    $P src/ms_dl.py Qwen/Qwen3-14B /data/public_models/Qwen3-14B
    touch /data/neg_opinion/outputs/.dl_done
  ' > outputs/download.log 2>&1 < /dev/null &
fi

# --- 3. HF 环境 ---
st "install HF stack (torch/transformers/peft/...)"
$VENV/bin/pip install $M torch transformers peft accelerate bitsandbytes datasets sentence-transformers bert-score scipy >> $LOG 2>&1
st "HF stack: $($VENV/bin/python -c 'import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.is_available())' 2>&1 | tail -1)"

# --- 4. vLLM 环境 ---
st "install vllm (accel venv)"
$ACC/bin/pip install -q -U pip $M >> $LOG 2>&1
$ACC/bin/pip install $M vllm >> $LOG 2>&1
st "vllm: $($ACC/bin/python -c 'import vllm,torch;print(vllm.__version__,torch.__version__)' 2>&1 | tail -1)"

# --- 5. 等下载完成 ---
st "wait downloads"
for i in $(seq 1 360); do [ -f outputs/.dl_done ] && break; sleep 30; done
st "Qwen3-14B: $(du -sh /data/public_models/Qwen3-14B 2>/dev/null | cut -f1) / $(ls /data/public_models/Qwen3-14B 2>/dev/null | wc -l) files"

# --- 6. 冒烟 ---
st "smoke test"
$VENV/bin/python - <<'PYEOF' >> $LOG 2>&1
import torch, transformers, peft, scipy
import sentence_transformers, bert_score
print("HF OK", torch.__version__, transformers.__version__,
      "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PYEOF
$ACC/bin/python -c "import vllm,torch;print('VLLM OK',vllm.__version__,torch.__version__)" >> $LOG 2>&1
st "SETUP DONE"
touch outputs/SETUP_DONE
