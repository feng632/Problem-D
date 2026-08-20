# -*- coding: utf-8 -*-
"""fig_data_dist.png —— 数据分布:左:三类残损实例数柱状图;右:单图实例数直方图。
读原始 YOLO 检测格式标注 code/data/dataset_3713/labels/train/*.txt(训练集,
跟 03-model-analysis.tex 里 8038/3934/946/3158、单图 1~50、近半数单实例
这几个数字对应的就是训练集统计口径)。不读 dataset_3713_seg(那是 solve_q2.py
转成 seg 多边形格式后的副本,用于训练,统计口径跟原始检测标注一致,读哪个
结果一样,这里直接读体积更小的原始标注)。
"""
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
LABEL_DIR = ROOT / "data" / "dataset_3713" / "labels" / "train"

CLASS_NAMES = ["Dent(凹陷)", "Hole(破洞)", "Rusty(锈蚀)"]
CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def load_train_labels():
    cls_count = {0: 0, 1: 0, 2: 0}
    per_img = []
    for f in sorted(LABEL_DIR.glob("*.txt")):
        lines = [ln.split() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        per_img.append(len(lines))
        for c, *_ in lines:
            cls_count[int(c)] += 1
    return cls_count, per_img


def main():
    if not LABEL_DIR.exists():
        print(f"[fig_data_dist] 找不到 {LABEL_DIR},请确认数据集已放入 code/data/dataset_3713/")
        sys.exit(1)

    cls_count, per_img = load_train_labels()
    n_img = len(per_img)
    if n_img == 0:
        print(f"[fig_data_dist] {LABEL_DIR} 下没有标注文件")
        sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- 左:三类残损实例数 ----
    counts = [cls_count[0], cls_count[1], cls_count[2]]
    bars = axes[0].bar(CLASS_NAMES, counts, color=CLASS_COLORS)
    axes[0].set_ylabel("实例数")
    axes[0].set_title(f"三类残损实例数(训练集,共 {sum(counts)} 个)")
    for b, c in zip(bars, counts):
        axes[0].text(b.get_x() + b.get_width() / 2, c + max(counts) * 0.01, str(c),
                     ha="center", fontsize=9)

    # ---- 右:单图实例数直方图 ----
    max_n = max(per_img)
    bins = range(1, max_n + 2)
    axes[1].hist(per_img, bins=bins, color="#4C72B0", edgecolor="white", align="left")
    frac_1 = sum(1 for x in per_img if x == 1) / n_img
    axes[1].set_xlabel("单图实例数")
    axes[1].set_ylabel("图像数")
    axes[1].set_title(f"单图实例数分布(训练集 N={n_img},单实例占比 {frac_1:.1%})")
    axes[1].axvline(1, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    fig.tight_layout()
    out_path = OUT / "fig_data_dist.png"
    fig.savefig(out_path, dpi=200)
    print(f"[fig_data_dist] -> {out_path}")
    print(f"[fig_data_dist] 三类实例数: Dent={counts[0]} Hole={counts[1]} Rusty={counts[2]} 合计={sum(counts)}")
    print(f"[fig_data_dist] 单图实例数: N={n_img} min={min(per_img)} max={max_n} 单实例占比={frac_1:.4f}")


if __name__ == "__main__":
    main()
