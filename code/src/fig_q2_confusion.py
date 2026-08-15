# -*- coding: utf-8 -*-
"""fig_q2_confusion.png —— 混淆矩阵热力图(3 类 + 背景,读 q2_result.pkl)。"""
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RESULT_PKL = ROOT / "data" / "q2_result.pkl"


def main():
    if not RESULT_PKL.exists():
        print(f"[fig_q2_confusion] 找不到 {RESULT_PKL},请先运行 solve_q2.py")
        sys.exit(1)
    d = joblib.load(RESULT_PKL)
    cm = np.array(d["confusion"])
    class_names = d.get("class_names", {0: "Dent", 1: "Hole", 2: "Rusty"})
    labels = [class_names[c] for c in sorted(class_names)] + ["背景(漏检/误检)"]

    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"问题二 混淆矩阵(3 类 + 背景,官方验证集,mode={d.get('mode')})")

    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > vmax * 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = OUT / "fig_q2_confusion.png"
    fig.savefig(out_path, dpi=200)
    print(f"[fig_q2_confusion] -> {out_path}")


if __name__ == "__main__":
    main()
