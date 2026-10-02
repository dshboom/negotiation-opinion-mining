---
name: 实验提案
about: 提出一个新实验（含唯一变量、对照组、验收标准）
title: "[实验] "
labels: experiment
assignees: ''
---

## 一句话假设
<!-- 你认为：做了 X，会导致 Y 提升/下降 -->

## 动机与依据
<!-- 关联的发现、诊断结论，或"哪些实验已经排除过相近思路"（可引用 EXPERIMENTS.md 的编号） -->

## 实验设计

| 项目 | 内容 |
|---|---|
| **唯一变量** | （一次只改一个东西） |
| **对照组** | |
| 基座模型 | Qwen3-14B / Qwen3-32B |
| 训练配置 | epochs / lora_r / lora_alpha / lr / grad_accum / max_len |
| 推理配置 | 引擎(HF/vLLM) / 温度 / 采样数 / max_new_tokens |
| 数据范围 | train 全量 / 子集；是否使用 calib |

## 验收标准
- [ ] 在 **val（300 条）** 上评测，与基线同口径
- [ ] 相对基线提升 > ______（写明阈值，建议 ≥ +0.01）
- [ ] 通过后才允许应用于 test

## 预期收益 / 风险
<!-- 期望涨多少？最可能失败的原因？ -->

## 相关脚本
<!-- 例如 src/train_lora_v2.py, scripts/phase2.sh -->
