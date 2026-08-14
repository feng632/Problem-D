# -*- coding: utf-8 -*-
"""示例画图脚本:保证 make fig 链路可跑通。

新题开始后删掉本文件,换成自己的 *_fig.py。画图脚本命名 fig_<内容>.py,
make fig 会运行 code/src/ 下 Makefile 中列出的所有 *_fig.py,并把
code/figures/ 的图同步到 paper/figures/。
"""
import matplotlib

matplotlib.use("Agg")  # 无显示环境下保存图片

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures"

x = np.linspace(0, 1, 100)
y = x**2

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(x, y)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Example Figure")
fig.tight_layout()
fig.savefig(OUT / "fig_example.png", dpi=200)
print("[fig_example] ->", OUT / "fig_example.png")
