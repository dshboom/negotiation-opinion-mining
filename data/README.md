# 数据说明

本仓库**不包含完整比赛数据**（请通过赛事方获取），仅提供 `sample_val.jsonl`（2 条）用于展示格式。

## 格式

每行一个 JSON 对象：

```json
{
  "sample_id": "SAMPLE_0001",
  "docs": [ { "doc_id": "DOC_0001", "doc_type": "国际新闻报道",
              "publish_date": "2025-07-26", "full_text": "……" } ],
  "issue_list": [
    { "issue_name": "……", "stance": "support|oppose|neutral",
      "argument_chain": ["原文逐字句子", "……"] }
  ],
  "future_argument": ["与每张议题卡一一对应的未来表态预测（80~120字）"]
}
```

## 本地数据布局（脚本默认路径）

```
data/
├── train.jsonl       # 2400 条 (有标签)
├── val.jsonl         #  300 条 (有标签)
├── test.jsonl        #  300 条 (无标签)
├── sft_train.jsonl   # 由 build_sft.py 生成
├── sft_val.jsonl
├── sft_test.jsonl
├── sft_calib.jsonl   # train 前 400 条 (校准用)
└── calib_gold.jsonl
```
