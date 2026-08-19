# -*- coding: utf-8 -*-
"""fig_q3_robust.png —— 各扰动下 mAP@0.5 柱状图(含基准)。
读 q3_result.pkl,不重新跑模型。"""
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
RESULT_PKL = ROOT / "data" / "q3_result.pkl"


def main():
    if not RESULT_PKL.exists():
        print(f"[fig_q3_robust] 找不到 {RESULT_PKL},请先运行 solve_q3.py")
        sys.exit(1)
    d = joblib.load(RESULT_PKL)
    rows = d["robustness"]
    names = [r["perturb"] for r in rows]
    maps = [r["mAP50"] for r in rows]
    deltas = [r["delta_map_pct"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#4C72B0"] + ["#DD8452"] * (len(names) - 1)
    bars = ax.bar(names, maps, color=colors)
    for bar, dv in zip(bars, deltas):
        label = "基准" if dv == 0.0 else f"Δ={dv:+.1f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, label,
                 ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(0, max(maps) * 1.25 if maps else 1.0)
    ax.set_title(f"问题三 扰动鲁棒性(官方验证集,N={d['n_test']},mode={d.get('mode')})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = OUT / "fig_q3_robust.png"
    fig.savefig(out_path, dpi=200)
    print(f"[fig_q3_robust] -> {out_path}")


if __name__ == "__main__":
    main()
