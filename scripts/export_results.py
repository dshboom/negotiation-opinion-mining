#!/usr/bin/env python3
"""
结果文件导出器 —— 由 docs/data/results/datasets.json 生成网页所需的：

  docs/data/results/index.json          ← 网页读取的聚合清单（含 viz 配置）
  docs/data/results/<id>.json           ← 单个数据集，供下载
  docs/data/results/<id>.csv            ← 单个数据集，供下载（Excel 可直接打开）

用法：
  python scripts/export_results.py            # 导出
  python scripts/export_results.py --check    # 只校验，不写文件（CI 可用）
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "data", "results", "datasets.json")
OUT = os.path.dirname(SRC)


def load():
    with open(SRC, encoding="utf-8") as f:
        return json.load(f)


def to_csv(ds):
    cols = ds["columns"]
    lines = []
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([c["label"] for c in cols])
    for row in ds.get("rows", []):
        vals = []
        for c in cols:
            v = row.get(c["key"])
            vals.append("" if v is None else v)
        w.writerow(vals)
    return buf.getvalue()


def validate(d):
    errs = []
    ids = set()
    for ds in d.get("datasets", []):
        if ds["id"] in ids:
            errs.append(f"重复数据集 id: {ds['id']}")
        ids.add(ds["id"])
        keys = {c["key"] for c in ds["columns"]}
        for i, row in enumerate(ds.get("rows", [])):
            extra = set(row) - keys
            if extra:
                errs.append(f"{ds['id']} 第 {i+1} 行含未定义列: {sorted(extra)}")
        for viz in ds.get("viz", []):
            if viz.get("type") == "bar":
                for k in ("labelKey", "valueKey"):
                    if viz.get(k) not in keys:
                        errs.append(f"{ds['id']} 图表引用了不存在的列: {viz.get(k)}")
    return errs


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    a = ap.parse_args()

    d = load()
    errs = validate(d)
    if errs:
        print("校验失败：")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    total_rows = sum(len(ds.get("rows", [])) for ds in d["datasets"])
    print(f"校验通过：{len(d['datasets'])} 个数据集 / {total_rows} 行 ✅")

    if a.check:
        return

    # 1) 聚合清单（网页用）
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

    # 2) 逐数据集 JSON + CSV（下载用）
    for ds in d["datasets"]:
        single = {k: v for k, v in ds.items()}
        with open(os.path.join(OUT, f"{ds['id']}.json"), "w", encoding="utf-8") as f:
            json.dump(single, f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUT, f"{ds['id']}.csv"), "w", encoding="utf-8-sig", newline="") as f:
            f.write(to_csv(ds))
        print(f"  → {ds['id']}.json / {ds['id']}.csv")

    print(f"完成，输出目录：{os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
