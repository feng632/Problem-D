# -*- coding: utf-8 -*-
"""fig_q3_sens.png —— 左:τ_nms -> mAP;右:分辨率 -> mAP/FPS(双轴)。
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
        print(f"[fig_q3_sens] 找不到 {RESULT_PKL},请先运行 solve_q3.py")
        sys.exit(1)
    d = joblib.load(RESULT_PKL)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # ---- 左:τ_nms -> mAP ----
    nms = d["nms_sensitivity"]
    x_nms = [r["iou_nms"] for r in nms]
    y_nms = [r["mAP50"] for r in nms]
    axes[0].plot(x_nms, y_nms, marker="o")
    axes[0].set_xlabel(r"NMS 阈值 $\tau_{nms}$")
    axes[0].set_ylabel("mAP@0.5")
    axes[0].set_title("NMS 阈值灵敏度")
    axes[0].grid(alpha=0.3)

    # ---- 右:分辨率 -> mAP(左轴) / FPS(右轴)双轴 ----
    res = d["resolution_sensitivity"]
    x_res = [r["imgsz"] for r in res]
    y_map = [r["mAP50"] for r in res]
    # fps_deploy: 部署阈值(conf_star)下单独计时的速度,才是真实部署会看到的数字;
    # fps_eval001(mAP 计算用的 conf=0.001 口径,候选框多、NMS 被放大)不画,只留在 pkl 里备查。
    conf_star = d["baseline"]["conf_star"]
    y_fps = [r.get("fps_deploy", r.get("fps")) for r in res]

    ax1 = axes[1]
    l1, = ax1.plot(x_res, y_map, marker="o", color="#4C72B0", label="mAP@0.5")
    ax1.set_xlabel("输入分辨率")
    ax1.set_ylabel("mAP@0.5", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.set_xticks(x_res)

    ax2 = ax1.twinx()
    l2, = ax2.plot(x_res, y_fps, marker="s", color="#DD8452", label=f"FPS(部署阈值 conf={conf_star:.2f})")
    ax2.set_ylabel(f"FPS(conf={conf_star:.2f},含数据读取)", color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")

    ax1.set_title("输入分辨率灵敏度(mAP vs FPS)")
    ax1.legend(handles=[l1, l2], loc="best", fontsize=9)
    ax1.grid(alpha=0.3)

    fig.suptitle(f"问题三 超参灵敏度(官方验证集,N={d['n_test']},mode={d.get('mode')})")
    fig.tight_layout()
    out_path = OUT / "fig_q3_sens.png"
    fig.savefig(out_path, dpi=200)
    print(f"[fig_q3_sens] -> {out_path}")


if __name__ == "__main__":
    main()
