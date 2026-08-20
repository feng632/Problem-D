# -*- coding: utf-8 -*-
"""fig_q2_pr.png —— 三类 PR 曲线(AP@0.5,读 q2_result.pkl,不重新训练)。"""
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RESULT_PKL = ROOT / "data" / "q2_result.pkl"


def main():
    if not RESULT_PKL.exists():
        print(f"[fig_q2_pr] 找不到 {RESULT_PKL},请先运行 solve_q2.py")
        sys.exit(1)
    d = joblib.load(RESULT_PKL)
    pr = d.get("pr_curves")
    class_names = d.get("class_names", {0: "Dent", 1: "Hole", 2: "Rusty"})
    metrics_table = {r["class"]: r for r in d.get("metrics_table", [])}

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for c, name in class_names.items():
        curve = pr.get(c) or pr.get(str(c))
        if not curve or not curve["recall"]:
            continue
        ap50 = metrics_table.get(name, {}).get("AP50", 0.0)
        ax.plot(curve["recall"], curve["precision"], label=f"{name} (AP@0.5={ap50:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"问题二 三类 PR 曲线(官方验证集,mode={d.get('mode')})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = OUT / "fig_q2_pr.png"
    fig.savefig(out_path, dpi=200)
    print(f"[fig_q2_pr] -> {out_path}")


if __name__ == "__main__":
    main()
