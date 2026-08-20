# -*- coding: utf-8 -*-
"""fig_data_scale.png —— 边界框面积占比分布(对数轴),训练集。
读原始 YOLO 检测格式标注 code/data/dataset_3713/labels/train/*.txt,
w*h(归一化宽高之积)即为该框占图像面积的比例。

注意(给论文手,别删):03-model-analysis.tex 第 10 行写"最大可达约 14%",
但训练集边界框面积占比的真实最大值是 99.688%(全画面),第 90 百分位才是
14.202%——数值上跟文字描述的"14%"几乎精确对上。判断是文字把"第90百分位"
写成了"最大",而不是数据本身有问题:排查发现 541 个(占 8038 个实例的
6.73%)边界框面积占比 >20%,三类都有但集中在 Dent(凹陷,477 个),Hole
25 个、Rusty 39 个;其中多个不同图像的 Dent 框坐标完全相同(w=h=0.998438,
即近乎整幅画面),不是随机噪声形态,更像是"大范围/弥漫性凹陷"标注惯例
(标注整个可见受损表面),不是标注错误——但这只是基于坐标分布形态的合理
推测,没有逐张目视核实。这张图按真实分布画
(对数轴,不裁尾),图上标出中位数与 p90 两条参考线,p90 数值会跟文字描述
的"14%"对上,但柱状图会诚实地把长尾画出来,跟文字并排放的时候这处矛盾
会很显眼——建议论文手/建模手对齐一下,要么把文字改成"第90百分位约14%,
个别弥漫性凹陷标注可达整幅画面",要么单独核实/处理这批 541 个异常框。
"""
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
LABEL_DIR = ROOT / "data" / "dataset_3713" / "labels" / "train"
CLASS_NAMES = {0: "Dent", 1: "Hole", 2: "Rusty"}


def load_areas():
    areas = []
    cls_of = []
    for f in sorted(LABEL_DIR.glob("*.txt")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            c, xc, yc, w, h = ln.split()
            areas.append(float(w) * float(h) * 100)  # 占图像面积的百分比
            cls_of.append(int(c))
    return np.array(areas), np.array(cls_of)


def main():
    if not LABEL_DIR.exists():
        print(f"[fig_data_scale] 找不到 {LABEL_DIR},请确认数据集已放入 code/data/dataset_3713/")
        sys.exit(1)

    areas, cls_of = load_areas()
    if len(areas) == 0:
        print(f"[fig_data_scale] {LABEL_DIR} 下没有标注文件")
        sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)

    median = float(np.median(areas))
    p90 = float(np.percentile(areas, 90))
    amax = float(np.max(areas))
    n_over20 = int((areas > 20).sum())

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins = np.logspace(np.log10(max(areas.min(), 1e-3)), np.log10(100), 40)
    ax.hist(areas, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("边界框面积占比(%,对数轴)")
    ax.set_ylabel("实例数")
    ax.set_title(f"边界框面积占比分布(训练集,N={len(areas)})")

    ax.axvline(median, color="#55A868", linestyle="--", linewidth=1.3)
    ax.axvline(p90, color="#C44E52", linestyle="--", linewidth=1.3)
    ymax = ax.get_ylim()[1]
    ax.text(median, ymax * 0.72, f"中位数={median:.2f}%", color="#55A868", fontsize=8, ha="left")
    ax.text(p90, ymax * 0.62, f"p90={p90:.2f}%", color="#C44E52", fontsize=8, ha="left")
    ax.text(0.98, 0.98, f"最大值={amax:.1f}%\n>20% 的框数={n_over20}(以 Dent 类为主)",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))

    fig.tight_layout()
    out_path = OUT / "fig_data_scale.png"
    fig.savefig(out_path, dpi=200)
    print(f"[fig_data_scale] -> {out_path}")

    p10 = float(np.percentile(areas, 10))
    print(f"[fig_data_scale] median={median:.4f}% p10={p10:.4f}% p90={p90:.4f}% max={amax:.4f}%")
    print(f"[fig_data_scale] >20% 的框数={n_over20}/{len(areas)}({n_over20 / len(areas):.2%}),"
          f"类别分布: {dict(zip(*np.unique(cls_of[areas > 20], return_counts=True)))}")
    print("[fig_data_scale] 注意:03-model-analysis.tex 写的\"最大可达约14%\"跟真实 max(99.688%附近)对不上,"
          "但跟 p90 几乎精确匹配,很可能是文字把 p90 误写成 max,详见脚本头注释与 noted_data.md。")


if __name__ == "__main__":
    main()
