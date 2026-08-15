# -*- coding: utf-8 -*-
"""fig_q2_loss.png —— 训练/验证损失收敛曲线(读 q2_result.pkl,不重新训练)。"""
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
        print(f"[fig_q2_loss] 找不到 {RESULT_PKL},请先运行 solve_q2.py")
        sys.exit(1)
    d = joblib.load(RESULT_PKL)
    lc = d.get("loss_curve")
    if not lc:
        print("[fig_q2_loss] q2_result.pkl 中没有 loss_curve 数据")
        sys.exit(1)

    epochs = lc["epoch"]
    train_cols = [c for c in lc if c.startswith("train/") and c.endswith("_loss")]
    val_cols = [c for c in lc if c.startswith("val/") and c.endswith("_loss")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for c in train_cols:
        axes[0].plot(epochs, lc[c], label=c.replace("train/", ""))
    axes[0].set_title("训练损失 (train)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for c in val_cols:
        axes[1].plot(epochs, lc[c], label=c.replace("val/", ""))
    axes[1].set_title("验证损失 (training-period val)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"问题二 YOLOv8-seg 训练损失收敛曲线(mode={d.get('mode')}, epochs={d.get('epochs')})")
    fig.tight_layout()
    out_path = OUT / "fig_q2_loss.png"
    fig.savefig(out_path, dpi=200)
    print(f"[fig_q2_loss] -> {out_path}")


if __name__ == "__main__":
    main()
