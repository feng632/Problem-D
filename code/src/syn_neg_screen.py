"""
问题一:合成负样本 —— 二次筛选(方案2 天空启发式 + 方案3 黑名单 + 方案4 分级)

背景:syn_neg_rules.py 生成的 tune/eval 里,已知混进两类漏网样本,且两类都跟
"拿检测模型自己筛自己"这条路(方案1)冲突——tune/eval 就是用来验证检测模型在
"确定无残损"图上会不会误触发的,拿被验证对象本身去筛选验证集是循环论证,所以
只用跟模型完全独立的内容特征筛,信号在真实案例上验证过再定阈值,不是拍脑袋:

1. 天空/低信息量误判(check_patch/detect_blank_block 抓不住的那类):
   patch 里天空对比轮廓边缘拉高了灰度方差,骗过两层方差过滤器,但天空本身在
   HSV 空间里有典型特征(低饱和度、高亮度),用连通域天空占比抓,不用灰度方差。
   已验证:181.jpg@(0,0)(459号问题块)算出 0.375,三个已确认正常的纹理块
   (3117.jpg两处喷漆面板、2950.jpg暗光纹理)都在 0.06 以下,信号干净。

2. 源图黑名单(方案3 的降级版):
   本来想找一个跟检测器无关的"可疑源图"自动判据(分辨率/宽高比/颜色统计/
   水印周期性 FFT),四种都在真实数据上测过——dataset_3713 里全部图片统一
   resize 到 640x640,分辨率/宽高比没有区分度;颜色统计、FFT周期性在已知
   问题源图(2905/511/3260)和正常样本之间也没有可用的分界。这类"图里有几个
   集装箱、标注全不全"的判断本质是语义理解,普通图像统计量摸不到,不硬凑一个
   假装能用的阈值。改成人工黑名单:已确认有问题的源图 stem 记在这里,
   人工审查发现新的随时加,拼图只要用到黑名单里的 stem,直接判最高怀疑级。

3. 分级:不再区分 tune/eval,1000 张放一个池子里,按怀疑分排三级,存到三个
   文件夹,文件名用分数打头方便按怀疑度排序审查。
   等你人工把三个文件夹里能用的挑出来,再二分(tune/eval)、重新编号——
   那一步等你筛完实际剩多少张再写,现在数量未知先不做。

用法:
    python syn_neg_screen.py
输出:
    code/data/syn_neg_screen/tier1_high/  {score:.3f}_{orig_index:04d}.jpg
    code/data/syn_neg_screen/tier2_mid/   同上
    code/data/syn_neg_screen/tier3_low/   同上
    code/data/syn_neg_screen/manifest.csv 每张图的可追溯信息(源图/坐标/分数/判定原因)
"""
from pathlib import Path
import csv
import random
import shutil

import cv2
import numpy as np

from syn_neg_rules import (
    IMG_PATH, OUT_DIR, CELL, TOTAL, SEED,
    get_all_cells, flatten_cells,
)

SCREEN_DIR = Path(__file__).parent.parent / "data" / "syn_neg_screen"

# 人工黑名单:已确认有问题的源图 stem。
# - 2905:多集装箱堆叠库存图,2905.txt 只标了一个 Hole,右侧另一个集装箱有
#   明显未标注的挤压破损。
# - 284:dreamstime.com 水印库存图,背景是几十个集装箱的堆叠,前景箱体有一个
#   明显破洞。题目原文(problem_D.txt)写明"验证集中每张图片集装箱的破损按照
#   严重程度从大到小最多取4个"——也就是说标注数量本来就有上限,背景堆叠里
#   哪怕真有没达到"前4严重"的破损,也不会被标出来,这是竞赛规则决定的结构性
#   风险,不是标注失误。此类"背景里一大堆集装箱"的库存图,风险天然更高。
# 后续人工审查发现新的问题源图,往这个 set 里加就行。
SOURCE_BLOCKLIST = {"2905", "284"}

SKY_S_THRESH = 60      # HSV 饱和度低于此值:判定为"天空色"候选像素
SKY_V_THRESH = 140     # HSV 亮度高于此值:判定为"天空色"候选像素
SKY_HUE_LOW = 85       # 天空典型色相下限(蓝青色调)
SKY_HUE_HIGH = 140     # 天空典型色相上限
SKY_NEAR_WHITE_S = 15  # 饱和度低于此值:基本无色相信息(过曝/阴天白色调),
                       # 不看色相也算数,避免漏掉纯白/过曝的天空
TIER1_SKY_FRAC = 0.5   # 天空连通域占比超过此值:大概率整块基本是天空,直接高危
TIER2_SKY_FRAC = 0.15  # 超过此值但不到 TIER1:中等怀疑,人工优先看


def sky_frac(patch: np.ndarray) -> float:
    """patch 内最大天空色连通域占比。用连通域而不是像素总占比,是为了避免
    "画面里散落着几个反光高光点"被误算成大片天空——只有连成一整片的才算数。

    光凭"低饱和度+高亮度"不够——浅色调的集装箱喷漆波纹板(反光、发白发粉)
    也会满足这个条件,实测会把正常波纹板误判成天空(frac=0.5+)。真天空和
    误判波纹板的色相分布截然不同:天空色相集中在蓝青调(H约100-120),
    误判案例集中在品红/粉调(H约140-180)。加一条色相范围限制,饱和度
    很低(接近无色,过曝/阴天白天空)时色相本身没意义,不卡这条。
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    color_ok = ((h >= SKY_HUE_LOW) & (h <= SKY_HUE_HIGH)) | (s < SKY_NEAR_WHITE_S)
    sky_mask = ((s < SKY_S_THRESH) & (v > SKY_V_THRESH) & color_ok).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sky_mask, connectivity=4)
    if n <= 1:
        return 0.0
    largest = stats[1:, cv2.CC_STAT_AREA].max()
    return float(largest / (patch.shape[0] * patch.shape[1]))


def rebuild_index_mapping() -> list[list[tuple[str, int, int]]]:
    """重新走一遍 syn_neg_rules.make_imgs 里同样的候选池构建 + 打乱逻辑
    (同样的 SEED,同样的 get_all_cells 扫描结果),精确还原出当前磁盘上
    每张 syn_neg_{i:04d}.jpg 分别用了源图的哪 4 个 (stem, x, y)。
    不重新读一遍磁盘图片内容,只重建"这张图是怎么拼出来的"这份账。
    """
    cells = get_all_cells()
    flat = flatten_cells(cells)
    random.seed(SEED)
    random.shuffle(flat)

    mapping = []
    idx = 0
    for _ in range(TOTAL):
        batch = flat[idx:idx + 4]
        if len(batch) < 4:
            break
        idx += 4
        mapping.append(batch)
    return mapping


def current_path(i: int) -> Path:
    split = "tune" if i < TOTAL // 2 else "eval"
    return OUT_DIR / split / f"syn_neg_{i:04d}.jpg"


def screen() -> None:
    mapping = rebuild_index_mapping()
    print(f"重建了 {len(mapping)} 张图的拼图账本")

    for tier in ("tier1_high", "tier2_mid", "tier3_low"):
        (SCREEN_DIR / tier).mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {"tier1_high": 0, "tier2_mid": 0, "tier3_low": 0}

    for i, batch in enumerate(mapping):
        src_path = current_path(i)
        if not src_path.exists():
            print(f"警告:{src_path} 不存在,跳过")
            continue

        stems = [stem for stem, _, _ in batch]
        blocklist_hit = any(stem in SOURCE_BLOCKLIST for stem in stems)

        max_sky = 0.0
        for stem, x, y in batch:
            img = cv2.imread(str(IMG_PATH / (stem + ".jpg")))
            patch = img[y:y + CELL, x:x + CELL]
            max_sky = max(max_sky, sky_frac(patch))

        if blocklist_hit:
            tier = "tier1_high"
            score = 1.0
            reason = "source_blocklist"
        elif max_sky >= TIER1_SKY_FRAC:
            tier = "tier1_high"
            score = max_sky
            reason = "sky_frac_high"
        elif max_sky >= TIER2_SKY_FRAC:
            tier = "tier2_mid"
            score = max_sky
            reason = "sky_frac_mid"
        else:
            tier = "tier3_low"
            score = max_sky
            reason = "clean"

        dst_name = f"{score:.3f}_{i:04d}.jpg"
        dst_path = SCREEN_DIR / tier / dst_name
        shutil.copy2(src_path, dst_path)
        counts[tier] += 1

        rows.append({
            "orig_index": i, "orig_path": str(src_path), "tier": tier,
            "score": f"{score:.3f}", "reason": reason,
            "max_sky_frac": f"{max_sky:.3f}", "blocklist_hit": blocklist_hit,
            "stems": ";".join(f"{stem}:{x},{y}" for stem, x, y in batch),
        })

    manifest_path = SCREEN_DIR / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"tier1_high(高危): {counts['tier1_high']} 张")
    print(f"tier2_mid (中危): {counts['tier2_mid']} 张")
    print(f"tier3_low (低危): {counts['tier3_low']} 张")
    print(f"manifest 存至 {manifest_path}")


if __name__ == "__main__":
    screen()
