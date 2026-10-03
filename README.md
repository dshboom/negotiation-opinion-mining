# 谈判观点挖掘 · Negotiation Opinion Mining

> 用 Qwen3 对**外交 / 时政长文本**做结构化观点挖掘：抽取多议题、判定立场、引用原文证据，并预测未来表态。
>
> 本仓库包含完整的**可复现流水线**、**1:1 复刻的官方评分器**、以及全部实验记录（含证伪项）。
>
> 🌐 **实时进度站点（GitHub Pages）：https://dshboom.github.io/negotiation-opinion-mining/**
> 数据源：`docs/data/progress.json`，日更新一条命令：`python scripts/update_progress.py --phase "..." --status doing --progress 40`

> **双3090专项进度（2026-10-03）**：[`实验动机、配置、实际进度与初筛成绩`](docs/DUAL3090_PROGRESS.md)。已完成四组40步训练，S01/S02的60篇初筛综合分为0.4330/0.4049；其余结果待评测。仅为短跑探索，不与历史全量结果混算。
> [双3090进度页面](https://dshboom.github.io/negotiation-opinion-mining/dual3090.html) · [汇总JSON](docs/data/dual3090.json)

---

## 当前研究主线与进度

**2026-10-03 12:32（中国标准时间）核验：** L20证据先行抽取SFT已完成（2000条、2轮、250步，约2小时23分）；联合SFT公平对照训练到155/250步（62%）；独立未来生成排队。新结构模型尚无val成绩，DPO尚未启动。

本次重建14B校准基线已全量评分：**综合0.5688、抽取0.6096、未来0.4054、F1 0.7387**。下方主结果表为历史实验，32B历史指标仍待原始报告核验，不与本次成绩混用。

路线：**结构化SFT → 训练内真实候选 → 偏好数据质检 → DPO/RFT比较 → 完整val验收**。优先抽取是因为综合评分80%来自抽取；设置同2000条fit的联合对照，避免把训练数据量变化误认为任务结构收益。

详细动机、逐阶段状态、划分和监控边界见 **[L20当前研究进度](docs/STRUCTURE_RUN_STATUS.md)**。GitHub展示的是已同步快照，不是秒级远端监控。原始预测、金标和适配器不公开。

---

## 任务定义

给定一篇文档（约 1500 字），输出一个 JSON：

```json
{
  "issue_list": [
    { "issue_name": "反对脱钩断链",
      "stance": "support",                              // support / oppose / neutral
      "argument_chain": ["原文逐字句子1", "原文逐字句子2"] }
  ],
  "future_argument": ["与第 1 张卡一一对应的未来表态预测（80~120 字）"]
}
```

- `argument_chain` 必须是**原文的逐字子串**；
- `future_argument` 条数必须与 `issue_list` **一一对应**；
- 一篇文档通常含 4~5 个议题。

---

## 主结果（验证集 300 篇）

综合分 = `0.8 × (F1 × α) + 0.2 × 预测得分`

| 配置 | 引擎 | F1 | α | 抽取(80%) | 未来(20%) | **综合** |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-14B 基座·零训练 | HF | 0.489 | 0.764 | 0.373 | 0.263 | 0.3511 |
| + SFT LoRA r16（2 轮） | HF | 0.713 | 0.825 | 0.588 | 0.398 | 0.5498 |
| + SFT LoRA r32（2 轮） | HF | 0.718 | 0.823 | 0.591 | 0.399 | 0.5527 |
| + vLLM 推理（无损加速 8.7×） | vLLM | 0.720 | 0.823 | 0.593 | 0.398 | 0.5538 |
| **+ ⑥ 立场校准（14B 最佳）** | HF | 0.737 | 0.825 | 0.608 | 0.408 | **0.5678** |
| **Qwen3-32B** | — | 0.773 | 0.849 | 0.665 | 0.380 | **0.6082** ★ |

- 参照上界：答案原样提交 = **1.0000**；整篇原文当证据 = **0.7309**（用于验证评分器与暴露规则特性）。

> 📦 **结果文件（JSON / CSV，可在线可视化 + 下载）**
> 站点 [「结果文件」板块](https://dshboom.github.io/negotiation-opinion-mining/#datasets) 按实验归档了 8 个数据集：主结果全指标、实验台账（E1–E13）、后处理（⑥⑦⑧）、Phase 0 归因与探针、Phase 1 负结果、vLLM 加速实测、评分字段字典。每个数据集都标注了**归属实验（E 编号）**。
> 文件位于 [`docs/data/results/`](docs/data/results/)（`.json` + `.csv` 成对）；修改 [`datasets.json`](docs/data/results/datasets.json) 后运行 `python scripts/export_results.py` 即可重新生成全部文件与网页清单。

---

## 核心发现

1. **不是"读不懂"，是"起名口径"不同** —— 金标偏爱**概括性主题词**（如「反对脱钩断链」），模型偏爱**贴原文**（如「APEC 开放区域主义」）。议题名占判定权重 40%，大量"近失卡"只差 0.05~0.10。
2. **证据早已逐字摘抄（99.2%）** —— 模型不需要"教它照抄"，真正的差距在**选哪一句、从哪开始截**。
3. **错误是系统性的，不是随机噪声** —— 同一题采样 5 次取共识（SC-merge）**反而更差**（−0.043），因为偏差每次都朝同一方向，投票无法纠正。

---

## 技术流水线

```
原始数据 ──build_sft.py──▶ 对话式 SFT 数据
                              │
                     train_lora.py / train_lora_v2.py   (LoRA 微调, completions-only)
                              │
                     infer_vllm.py                       (vLLM 推理, 8.7× 加速)
                              │
                     eval_scorer.py                      (官方评分逻辑 1:1 复刻)
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
postprocess.py(⑥⑦⑧)   fit_bias.py + apply_stance_calib.py   sc_merge.py
       │                    (⑥ 立场校准, 唯一有效)         (自一致性合并, 已证伪)
       └──────────────────────┴──────────────────────┘
                              │
                     result_final.jsonl (提交)
```

### 关键设计
- **只在答案上算 loss（completions-only）**，并按官方 `loss_scale=ignore_empty_think` 忽略空 think 块；
- **训练 / 推理统一 `enable_thinking=False`**；
- **val 验收制**：任何改动用有标签的 val 验证，通过才应用于 test；
- **断点续跑**：每个阶段有完成标记，训练可从 checkpoint 恢复。

---

## 实验记录

> 📒 **完整实验日志（含全部参数、结论与证伪原因）：[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)**
> 新实验请先用 [实验提案模板](.github/ISSUE_TEMPLATE/experiment_proposal.md) 开 Issue，完成后用 [实验记录模板](.github/ISSUE_TEMPLATE/experiment_log.md) 归档。

### ✅ 有效
| 实验 | 说明 | 收益 |
|---|---|---|
| SFT LoRA r16（2 轮） | 14B + LoRA，completions-only | **+0.1987** |
| SFT LoRA r32（2 轮） | 仅加大 LoRA 秩 | +0.0029（说明 rank 已饱和） |
| vLLM 推理 | PagedAttention + 连续批处理 | 生成 **8.7×**，任务级无损 |
| ⑥ 立场校准 | 无标签先验匹配，校准立场分布 | **+0.0151** |

### ❌ 证伪（同样重要，避免重复试错）
| 实验 | 结论 |
|---|---|
| ⑦ 卡片数截断 | 模型本就输出 4~5 张，无截断空间 |
| ⑧ 近重复去重 | 输出几乎无重复，去重反伤基线 |
| Liger 融合核 | 瓶颈在 CPU，无提升 |
| prompt 追加指令 | 分布外扰动，−0.018 |
| **SC-merge 自一致性合并** | **系统性偏差无法投票纠正，−0.043~−0.088** |

### 🔄 进行中
- **字段加权重训**：对 `issue_name` ×2、`stance` ×3、`future_argument` ×2 的 token 加权（自定义加权交叉熵 + 分块计算防 OOM）。

---

## 目录结构

```
.
├── src/                     # 核心算法脚本
│   ├── build_sft.py         #   竞赛数据 → 对话式 SFT 数据
│   ├── train_lora.py        #   LoRA 训练
│   ├── train_lora_v2.py     #   字段加权重训
│   ├── infer.py             #   HF 推理
│   ├── infer_vllm.py        #   vLLM 推理
│   ├── sample_vllm.py       #   多采样（自一致性实验）
│   ├── eval_scorer.py       #   官方评分逻辑复刻
│   ├── rescore.py           #   立场概率前向
│   ├── fit_bias.py          #   ⑥ 偏置拟合
│   ├── apply_stance_calib.py#   ⑥ 偏置应用
│   ├── postprocess.py       #   ⑥⑦⑧ 后处理矩阵
│   ├── sc_merge.py          #   自一致性合并
│   ├── diag_phase0.py       #   失败归因诊断
│   ├── analyze_names.py     #   议题命名口径分析
│   ├── export_pred.py       #   模型逐条输出 → 可读 CSV/JSON（含金标对照与失败原因）
│   ├── ms_dl.py             #   ModelScope 断点续传下载
│   └── pipeline.py          #   无人值守主流水线
├── scripts/                 # 编排 / 部署脚本
├── docs/                    # 报告与演示材料
└── data/                    # 数据说明 + 样例（完整数据需自行获取）
```

---

## 快速开始

### 1. 环境

```bash
# 训练 / 评测环境
pip install -r requirements.txt

# 推理加速环境（建议独立 venv；无 nvcc 的机器必须关闭 FlashInfer）
python -m venv .venv-accel && . .venv-accel/bin/activate
pip install vllm
export VLLM_USE_FLASHINFER_SAMPLER=0
```

> **国内网络**：模型走 **ModelScope（模搭）**下载；pip 建议使用清华镜像
> `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

### 2. 数据准备

```bash
# 将 train.jsonl / val.jsonl / test.jsonl 放入 data/
python src/build_sft.py --inp data/train.jsonl --out data/sft_train.jsonl --labeled
python src/build_sft.py --inp data/val.jsonl   --out data/sft_val.jsonl   --labeled
python src/build_sft.py --inp data/test.jsonl  --out data/sft_test.jsonl
head -400 data/sft_train.jsonl > data/sft_calib.jsonl   # 校准集(可选)
```

### 3. 训练

```bash
python src/train_lora.py --out outputs/qwen3-14b-lora-r32 \
    --epochs 2 --max-len 6144 --lora-r 32 --lora-alpha 64 \
    --lr 1e-4 --grad-accum 16
```

### 4. 推理与评测

```bash
# vLLM 推理
.venv-accel/bin/python src/infer_vllm.py --data data/sft_val.jsonl \
    --out eval/pred_r32_val.jsonl --adapter outputs/qwen3-14b-lora-r32

# 官方口径评分
python src/eval_scorer.py --gold data/val.jsonl --pred eval/pred_r32_val.jsonl \
    --device cuda --out eval/report_r32.json
```

### 5. ⑥ 立场校准

```bash
python src/rescore.py --data data/sft_val.jsonl --pred eval/pred_r32_val.jsonl \
    --out eval/rescore_r32_val.jsonl --adapter outputs/qwen3-14b-lora-r32
python src/fit_bias.py --rescore eval/rescore_r32_val.jsonl --out eval/bias.json
python src/apply_stance_calib.py --pred eval/pred_r32_val.jsonl \
    --rescore eval/rescore_r32_val.jsonl \
    --b-neutral 0.7 --b-oppose -0.5 --out eval/pred_r32_val_cal6.jsonl
```

### 6. 一键流水线

```bash
bash scripts/run_all.sh      # 幂等 + 断点续跑
```

---

## 评分规则（复刻口径）

```
语义等价分 = 0.4 × cos(议题名) + 0.6 × cos(证据链)      # bge-small-zh-v1.5
立场必须严格相等（硬门槛），一对一匹配（匈牙利算法），阈值 0.7 → NC
P = NC/Np    R = NC/Ng    F1 = 2PR/(P+R)
α = 匹配对的 BERTScore-F1 均值                          # bert-base-chinese
抽取分 = F1 × α                (占 80%)
未来分 = (ROUGE-L + BERTScore)/2  (占 20%)
综合分 = 0.8 × 抽取分 + 0.2 × 未来分
```

> 自检：把标准答案原样提交得 **1.0000**；把整篇原文当证据得 **0.7309**。

---

## 协作 / 贡献

- **实验提案**：用 [`实验提案模板`](.github/ISSUE_TEMPLATE/experiment_proposal.md) 开 Issue（含唯一变量、对照组、验收标准）。
- **实验归档**：完成后用 [`实验记录模板`](.github/ISSUE_TEMPLATE/experiment_log.md) 记录，并把结论补进 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。
- **缺陷 / 建议**：用 Bug 报告或功能建议模板。
- **铁律**：**任何改动先在 val（300 条）验收，通过才应用于 test**；一次只改一个变量。

---

## 说明

- **数据**：比赛数据版权归赛事方，本仓库**不含完整数据**，仅提供 `data/sample_val.jsonl`（2 条样例）。
- **模型**：Qwen3 系列模型请通过 [ModelScope](https://modelscope.cn) 获取。
- **合规**：正式提交按要求使用 **Qwen3-32B** 作为基座。
- 本仓库用于技术交流与复现，请遵守赛事相关规定。


## 双3090短跑探索（2026-10-03 12:31 CST）

详见 [动机、配置、实际进度和初筛结果](docs/DUAL3090_PROGRESS.md)

四组40步训练已完成；S01/S02固定60篇综合分0.4330/0.4049，证据先行字段顺序变体当前低于基线0.0281。S03预测中，S04待评测，S05训练29/40，S06待执行。短跑分数不是历史全量成绩；生成吞吐是当前瓶颈。
