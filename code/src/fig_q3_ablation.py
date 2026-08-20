# -*- coding: utf-8 -*-
"""fig_q3_ablation.png —— 消融实验(逐项累加)mAP@0.5 / AP_s 趋势图。
读 5 份结果:
  code/data/q3_ablate_step{0,1,2,3}.pkl                 (本脚本对应的消融专用训练,30ep)
  code/data/q3_explore_explore_s_b2_i1280_e30.pkl        (step4=+强增强,复用探索性训练,30ep)
不重新跑模型。同时把 tab:q3-ablation 需要的数值打印成可直接粘贴进
paper/sections/04-models.tex 表格骨架的形式(该文件本身不动 .tex)。

注意:这 5 步全部只跑了 30 epoch(消融实验的时间预算约束),数值远低于
solve_q2.py 官方 100-epoch 训练结果(q2_result.pkl,step4 同配置下
mAP@0.5=0.3623 vs 这里 30ep 只有 0.1828)——消融表只用于看"加/减某个设计
的相对增量",不能拿来跟主结果表的绝对数值对比,见 noted_q3.md。
"""
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
DATA_DIR = ROOT / "data"

STEP_LABELS = ["基准", "+类别加权", "+P2头", "+1280输入", "+强增强"]
STEP_FILES = [
    DATA_DIR / "q3_ablate_step0.pkl",
    DATA_DIR / "q3_ablate_step1.pkl",
    DATA_DIR / "q3_ablate_step2.pkl",
    DATA_DIR / "q3_ablate_step3.pkl",
    DATA_DIR / "q3_explore_explore_s_b2_i1280_e30.pkl",
]


def load_step(path):
    d = joblib.load(path)
    ev = d.get("eval_official")
    if not ev:
        return None
    pcm = ev["per_class_metrics"]
    ap50 = sum(v["AP50"] for v in pcm.values()) / len(pcm)
    ap_s = sum(v["AP_s"] for v in pcm.values()) / len(pcm)
    return ap50, ap_s


def main():
    missing = [p for p in STEP_FILES if not p.exists()]
    if missing:
        print(f"[fig_q3_ablation] 缺少结果文件: {missing}")
        print("[fig_q3_ablation] 请先跑 ablate_q3_steps.py --step {0,1,2,3},"
              "step4 需要 q3_explore_explore_s_b2_i1280_e30.pkl 且带 eval_official")
        sys.exit(1)

    rows = []
    for label, path in zip(STEP_LABELS, STEP_FILES):
        r = load_step(path)
        if r is None:
            print(f"[fig_q3_ablation] {path.name} 缺 eval_official,跳过")
            sys.exit(1)
        rows.append((label, *r))

    OUT.mkdir(parents=True, exist_ok=True)

    labels = [r[0] for r in rows]
    map50s = [r[1] for r in rows]
    ap_ss = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(labels))
    ax.plot(x, map50s, marker="o", color="#4C72B0", label="mAP@0.5")
    ax.plot(x, ap_ss, marker="s", color="#DD8452", label="$AP_s$")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("AP")
    ax.set_ylim(0, max(map50s + ap_ss) * 1.25)
    ax.set_title("问题三 消融实验(逐项累加,30 epoch,官方验证集)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for xi, y1, y2 in zip(x, map50s, ap_ss):
        ax.text(xi, y1 + 0.006, f"{y1:.4f}", ha="center", fontsize=7, color="#4C72B0")
        ax.text(xi, y2 - 0.014, f"{y2:.4f}", ha="center", fontsize=7, color="#DD8452")
    fig.tight_layout()

    out_path = OUT / "fig_q3_ablation.png"
    fig.savefig(out_path, dpi=200)
    print(f"[fig_q3_ablation] -> {out_path}")

    print("\n[fig_q3_ablation] tab:q3-ablation 可直接粘贴的表格行:")
    for label, map50, ap_s in rows:
        print(f"% {label:12s} & {map50:.4f} & {ap_s:.4f} \\\\")


if __name__ == "__main__":
    main()
