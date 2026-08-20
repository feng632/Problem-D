# -*- coding: utf-8 -*-
"""fig_q2_visual.png —— 4 张验证集样例:原图 + 预测框 + 预测掩码 + 类别标签叠加。

读取 q2_result.pkl 里 solve_q2.py 已经存好的原始预测(viz_samples),不重新跑模型。
展示时按置信度阈值 + top-4(按面积)做一次筛选,避免把 conf=0.001 评估口径下的
大量低分框全部画出来导致图不可读(全框口径的定量指标仍以 q2_result.pkl 中的
AP/Dice 等数值为准,这里只是可视化展示用的简化)。
"""
import sys
from pathlib import Path

import cv2
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

DISPLAY_CONF = 0.25
TOPK_DISPLAY = 4
COLORS = {0: (230, 25, 75), 1: (60, 180, 75), 2: (0, 130, 200)}  # BGR-ish for matplotlib RGB use as RGB below


def _select_for_display(pred, masks):
    idx = [i for i, p in enumerate(pred) if p["score"] >= DISPLAY_CONF]
    if not idx:
        # 置信度都很低(比如冒烟测试的欠训练模型)时退化为取分数最高的几个,保证图能画出来
        idx = list(np.argsort([-p["score"] for p in pred])[:TOPK_DISPLAY])
    else:
        areas = []
        for i in idx:
            b = pred[i]["box"]
            areas.append(max(b[2] - b[0], 0) * max(b[3] - b[1], 0))
        order = np.argsort(-np.array(areas))
        idx = [idx[o] for o in order][:TOPK_DISPLAY]
    return idx


def main():
    if not RESULT_PKL.exists():
        print(f"[fig_q2_visual] 找不到 {RESULT_PKL},请先运行 solve_q2.py")
        sys.exit(1)
    d = joblib.load(RESULT_PKL)
    samples = d.get("viz_samples", [])
    class_names = d.get("class_names", {0: "Dent", 1: "Hole", 2: "Rusty"})
    if not samples:
        print("[fig_q2_visual] q2_result.pkl 中没有 viz_samples")
        sys.exit(1)

    n = min(4, len(samples))
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
    if n == 1:
        axes = [axes]

    for ax, s in zip(axes, samples[:n]):
        img = cv2.imread(s["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay = img.copy()

        idx = _select_for_display(s["pred"], s["pred_masks"])
        for i in idx:
            p = s["pred"][i]
            mask = s["pred_masks"][i] if i < len(s["pred_masks"]) else None
            color = COLORS.get(p["cls"], (255, 255, 0))
            if mask is not None:
                overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array(color)).astype(np.uint8)

        ax.imshow(overlay)
        for i in idx:
            p = s["pred"][i]
            x1, y1, x2, y2 = p["box"]
            color = np.array(COLORS.get(p["cls"], (255, 255, 0))) / 255.0
            ax.add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
            )
            ax.text(
                x1,
                max(y1 - 4, 0),
                f"{class_names[p['cls']]} {p['score']:.2f}",
                color="white",
                fontsize=8,
                bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor="none"),
            )
        ax.set_title(s["stem"], fontsize=9)
        ax.axis("off")

    fig.suptitle(f"问题二 检测+分割可视化样例(预测,mode={d.get('mode')},conf>={DISPLAY_CONF} 或 top-{TOPK_DISPLAY})")
    fig.tight_layout()
    out_path = OUT / "fig_q2_visual.png"
    fig.savefig(out_path, dpi=200)
    print(f"[fig_q2_visual] -> {out_path}")


if __name__ == "__main__":
    main()
