#!/usr/bin/env python3
# =====================================================================
#  rename_to_prob.py — 把论文 PDF 重命名为国赛要求的命名格式
#
#  国赛要求:提交的论文 PDF 命名为  报名号-选题编号.pdf (如 1234567-B.pdf)
#  本脚本把 paper/main.pdf 复制为 submission/报名号-B.pdf
#
#  用法:
#    python rename_to_prob.py 1234567
#  (报名号从全国大学生数学建模竞赛官网报名系统获取)
# =====================================================================
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER_PDF = ROOT / "paper" / "main.pdf"
SUBM = ROOT / "submission"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    reg_no = sys.argv[1].strip()
    if not reg_no.isdigit():
        print(f"[错误] 报名号应为数字,收到: {reg_no!r}")
        sys.exit(1)

    if not PAPER_PDF.exists():
        print(f"[错误] 未找到论文 PDF: {PAPER_PDF}\n请先运行 make pdf")
        sys.exit(1)

    # 按报名号猜测选题:报名号末位是题目编号(A/B/C/D/E/F)
    prob = reg_no[-1]
    dest = SUBM / f"{reg_no}-{prob}.pdf"
    SUBM.mkdir(exist_ok=True)
    shutil.copy(PAPER_PDF, dest)
    print(f"[ok] 论文已复制为: {dest}")


if __name__ == "__main__":
    main()
