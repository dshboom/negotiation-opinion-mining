#!/usr/bin/env python3
"""
进度更新助手 —— 修改 docs/data/progress.json（GitHub Pages 站点的唯一数据源）。

用法示例：
  # 更新时间戳（每次必做）
  python scripts/update_progress.py --touch

  # 更新某个阶段
  python scripts/update_progress.py --phase "r32 基线重训（2 轮）" --status done --progress 100 \
      --detail "训练完成，loss 0.50"

  # 更新 KPI
  python scripts/update_progress.py --kpi "14B 最佳" "0.5678"

  # 更新最高分
  python scripts/update_progress.py --best 0.62 --best-label "新模型（val 300）"

  # 追加一行主结果（JSON）
  python scripts/update_progress.py --add-result '{"name":"字段加权重训","engine":"vLLM","f1":0.76,"alpha":0.84,"extract":0.64,"future":0.42,"composite":0.60,"mark":"实验"}'

  # 追加 / 更新待办
  python scripts/update_progress.py --todo "32B 应用 ⑥ 校准"

  # 实验状态（按 id）
  python scripts/update_progress.py --exp-status E11 ok --exp-gain "+0.012"

  # 查看当前摘要
  python scripts/update_progress.py --show
"""
import argparse, json, os, sys, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "docs", "data", "progress.json")


def load():
    with open(PATH, encoding="utf-8") as f:
        return json.load(f)


def save(d):
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print("已更新", PATH)


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--touch", action="store_true", help="只更新时间戳")
    ap.add_argument("--phase", help="阶段名（需与 progress.json 中一致）")
    ap.add_argument("--status", choices=["done", "doing", "pending"], help="阶段状态")
    ap.add_argument("--progress", type=int, help="阶段进度 0-100")
    ap.add_argument("--detail", help="阶段说明")
    ap.add_argument("--kpi", nargs=2, metavar=("LABEL", "VALUE"), help="更新 KPI")
    ap.add_argument("--best", help="更新最高分")
    ap.add_argument("--best-label", help="更新最高分说明")
    ap.add_argument("--add-result", help="追加主结果行的 JSON 字符串")
    ap.add_argument("--todo", help="追加待办（已存在则忽略）")
    ap.add_argument("--todo-status", choices=["done", "doing", "pending"], default="pending")
    ap.add_argument("--exp-status", nargs=2, metavar=("ID", "STATUS"), help="按 ID 设置实验状态")
    ap.add_argument("--exp-gain", nargs=2, metavar=("ID", "GAIN"), help="按 ID 设置实验收益")
    ap.add_argument("--show", action="store_true", help="打印当前摘要")
    a = ap.parse_args()

    d = load()

    if a.phase:
        hit = [p for p in d.get("phases", []) if p["name"] == a.phase]
        if not hit:
            print("!! 找不到阶段:", a.phase)
            print("   现有阶段:", [p["name"] for p in d.get("phases", [])])
            sys.exit(1)
        if a.status: hit[0]["status"] = a.status
        if a.progress is not None: hit[0]["progress"] = a.progress
        if a.detail: hit[0]["detail"] = a.detail

    if a.kpi:
        label, value = a.kpi
        for k in d.get("kpis", []):
            if k["label"] == label:
                k["value"] = value
                break
        else:
            d.setdefault("kpis", []).append({"label": label, "value": value})

    if a.best: d["best"] = a.best
    if a.best_label: d["best_label"] = a.best_label

    if a.add_result:
        d.setdefault("results", []).append(json.loads(a.add_result))

    if a.todo:
        if not any(t["task"] == a.todo for t in d.get("todos", [])):
            d.setdefault("todos", []).append({"task": a.todo, "status": a.todo_status})

    if a.exp_status:
        eid, st = a.exp_status
        for e in d.get("experiments", []):
            if e["id"] == eid:
                e["status"] = st
                break

    if a.exp_gain:
        eid, g = a.exp_gain
        for e in d.get("experiments", []):
            if e["id"] == eid:
                e["gain"] = g
                break

    d["updated"] = now()
    save(d)

    if a.show:
        print("\n== 摘要 ==")
        print("最佳:", d.get("best"), "|", d.get("best_label"))
        print("阶段:")
        for p in d.get("phases", []):
            print(f"  [{p['status']:7s}] {p['progress']:3d}%  {p['name']}  — {p.get('detail','')}")
        print("实验:", ", ".join(f"{e['id']}:{e['status']}" for e in d.get("experiments", [])))


if __name__ == "__main__":
    main()
