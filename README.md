# Problem-D — 数学建模竞赛团队仓库(模板)

基于 Problem-B 沉淀出的通用骨架:LaTeX 国赛模板 + Makefile 构建 + 三人协作分节结构。

## 起步
1. 本题发题后,把 `paper/main.tex` 的 `\title`、`\tihao`、团队信息改为本题内容
2. 按 `FILL_CHECKLIST.md` 建立脚本分工与回填清单(发题第一天写)
3. 数据下载到 `data/` 与 `code/data/`(不提交 git),读法写进 `data/README.md`
4. 开写:各节见 `paper/sections/`

## 常用命令(仓库根目录)
```
make fig     # 运行 code/src/*_fig.py → code/figures/ → 同步 paper/figures/
make pdf     # xelatex 编译 paper/main.pdf
make all     # fig → pdf → pack 一条龙
make pack    # submission/main.zip + submission/main.md5
```
