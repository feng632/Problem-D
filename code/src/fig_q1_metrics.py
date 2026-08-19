# -*- coding: utf-8 -*-
"""fig_q1_metrics.png —— 左:ROC 曲线(评估集);右:F1(τ) 曲线(调参集,标注 τ*)。
读 q1_result.pkl,不重新跑模型。"""
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
RESULT_PKL = ROOT / "data" / "q1_result.pkl"


def main():
    if not RESULT_PKL.exists():
        print(f"[fig_q1_metrics] 找不到 {RESULT_PKL},请先运行 solve_q1.py")
        sys.exit(1)
    d = joblib.load(RESULT_PKL)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # ---- 左:ROC(评估集) ----
    roc = d["roc"]
    auc_val = d["eval_metrics"]["auc"]
    axes[0].plot(roc["fpr"], roc["tpr"], label=f"ROC (AUC={auc_val:.4f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="随机猜测")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title(f"ROC 曲线(评估集,N={d['eval_n_pos'] + d['eval_n_neg']})")
    axes[0].legend(fontsize=9, loc="lower right")
    axes[0].grid(alpha=0.3)

    # ---- 右:F1(τ)(调参集,标注 τ*) ----
    tau_grid = d["tau_grid"]
    f1_tune = d["f1_tune"]
    tau_star = d["tau_star"]
    axes[1].plot(tau_grid, f1_tune, marker="o", markersize=3, label="F1(τ)")
    axes[1].axvline(tau_star, color="red", linestyle="--", linewidth=1, label=f"τ*={tau_star:.2f}")
    axes[1].set_xlabel("τ")
    axes[1].set_ylabel("F1")
    axes[1].set_xlim(0, 1)
    axes[1].set_title(f"F1(τ) 曲线(调参集,N={d['tune_n_pos'] + d['tune_n_neg']})")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    m = d["eval_metrics"]
    fig.suptitle(
        f"问题一 检测归约分类指标(mode={d.get('mode')})  "
        f"评估集@τ*: Acc={m['acc']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}"
    )
    fig.tight_layout()
    out_path = OUT / "fig_q1_metrics.png"
    fig.savefig(out_path, dpi=200)
    print(f"[fig_q1_metrics] -> {out_path}")


if __name__ == "__main__":
    main()
