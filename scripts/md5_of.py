#!/usr/bin/env python3
# =====================================================================
#  md5_of.py — 计算文件的 MD5 值,输出到指定文件或直接打印
#
#  用法:
#    python md5_of.py <输入文件>            # 打印 MD5 值
#    python md5_of.py <输入文件> <输出文件>  # 写入输出文件(带文件名,便于粘贴)
#
#  国赛说明:MD5 码是上传论文 PDF 文件时用的校验码,官网会给出算法说明,
#  一般是对 PDF 文件本身计算 MD5。注意核对官网要求的计算对象与输出格式。
# =====================================================================
import hashlib
import sys


def md5_of_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    src = sys.argv[1]
    digest = md5_of_file(src)

    if len(sys.argv) >= 3:
        # 写入文件:文件名 + MD5,方便直接粘贴
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            f.write(f"{src}  {digest}\n")
        print(digest)
    else:
        print(digest)


if __name__ == "__main__":
    main()
