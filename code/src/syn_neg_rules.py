"""
问题一:合成负样本 —— 选块规则(骨架)

已在 demo_free_cell_scan.py 里跑通验证过的核心逻辑,整理成骨架函数放这里,
供后续扩展成完整的合成负样本流水线。

规则:
- 一张残损图的标注框(YOLO格式,归一化 class x_center y_center w h)按 margin
  像素外扩后当作"禁区"(避开框边缘可能残留的框外残损)
- 用 cell x cell 的窗口、step 步长在图上滑动扫描
- 跟任意一个外扩框有重叠的窗口被淘汰,不重叠的记为"干净块"候选,只返回像素坐标

TODO(用户接着写):
1. 批量遍历 code/data/dataset_3713/images/train 全部图(只用 train,test 是最终
   验证集只能用一次,不能拿来裁块避免泄漏),对每张图调用 scan_free_cells,把
   (img_path, x, y, cell) 候选收集到一个大池子里
2. 从候选池里随机抽块,4 块拼成一张 640x640(或自己设计别的拼法),注意:
   - 同一张拼图里的 4 块最好来自不同源图,避免明显的纹理重复
   - 拼接边缘要不要做羽化/融合,自己权衡(不做也行,先检验能不能跑通)
3. 批量生成:目标共 800 张,调参半 400 / 评估半 400,存到
   code/data/syn_neg/{tune,eval}/
4. 抽检:
   - 自动规则筛一遍(比如拼接缝的像素梯度突变检测),先过滤掉明显有问题的
   - 再随机抽一小批人工看一眼,确认整体质量、没有框外残损混进来
5. 把上面这套接进 solve_q1.py,作为"合成负样本"步骤的产出
"""
from pathlib import Path
import random

import cv2
import numpy as np

DATA_PATH = Path(__file__).parent.parent / "data" / "dataset_3713"
IMG_PATH = DATA_PATH / "images" / "train"
LABEL_PATH = DATA_PATH / "labels" / "train"

OUT_DIR = Path(__file__).parent.parent / "data" / "syn_neg"
CELL = 320
CANVAS = 640
TOTAL = 1000
SEED = 42

VAR_THRESH = 200        # 整块方差低于此值:判定为纯色/天空一类,丢弃
BLANK_GRID = 8          # 局部打码检测:把 cell 切成 grid x grid 的小格子
FLAT_VAR_THRESH = 1.0   # 小格子局部方差低于此值:判定为"无纹理"(真实照片噪声/量化
                        # 几乎不可能让方差精确趋近 0,只有数字插入的纯色块会这样;
                        # 用它代替直接数 Canny 边缘像素,规避暗光图整体边缘稀疏导致的误判)
BLANK_MIN_FRAC = 0.08   # "无纹理"小格子连成一片,占比超过此值:判定为局部打码块


def load_boxes_px(label_path: Path, w: int, h: int, margin: int = 12) -> list[tuple[float, float, float, float]]:
    """读 YOLO 格式标注,转成像素坐标并按 margin 外扩。
    返回 [(x1,y1,x2,y2), ...],已外扩,可能超出图像边界(调用方按需 clip)。
    """
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        _, xc, yc, bw, bh = map(float, line.split())
        x1 = (xc - bw / 2) * w - margin
        y1 = (yc - bh / 2) * h - margin
        x2 = (xc + bw / 2) * w + margin
        y2 = (yc + bh / 2) * h + margin
        boxes.append((x1, y1, x2, y2))
    return boxes


def cell_is_free(cx1: float, cy1: float, cx2: float, cy2: float,
                  boxes: list[tuple[float, float, float, float]]) -> bool:
    """矩形 (cx1,cy1,cx2,cy2) 是否与 boxes 中任一(已外扩的)框有重叠。"""
    for (x1, y1, x2, y2) in boxes:
        if not (cx2 <= x1 or cx1 >= x2 or cy2 <= y1 or cy1 >= y2):
            return False
    return True

def scan_free_cells(img_w: int, img_h: int, label_path: Path,
                     cell: int = 320, step: int = 160, margin: int = 12) -> list[tuple[int, int]]:
    """滑动窗口扫描一张图,返回所有"干净块"的左上角坐标列表 [(x, y), ...]。
    只返回坐标,不读图像素——批量拼图时由调用方按坐标去原图裁剪。
    """
    boxes = load_boxes_px(label_path, img_w, img_h, margin=margin)
    free = []
    for y in range(0, img_h - cell + 1, step):
        for x in range(0, img_w - cell + 1, step):
            if cell_is_free(x, y, x + cell, y + cell, boxes):
                free.append((x, y))
    return free

def get_all_cells() -> list[tuple[str, list[tuple[int, int]]]]:
    """遍历 train 全部图,返回 [(stem, [(x,y), ...]), ...]。
    stem 是不带后缀的文件名(字符串),对应图片和标注共用的编号。
    """
    avail_cells = []
    for label in LABEL_PATH.glob("*.txt"):
        img_file_path = IMG_PATH / (label.stem + ".jpg")
        img = cv2.imread(str(img_file_path))
        if img is None:
            print(f"{label.stem} not found")
            continue
        h, w = img.shape[:2]
        cur_free_cells = scan_free_cells(w, h, label)
        valid_cells = [
            (x, y) for (x, y) in cur_free_cells
            if is_cell_valid(img[y:y + CELL, x:x + CELL])
        ]
        if valid_cells:
            avail_cells.append((label.stem, valid_cells))
    return avail_cells


def flatten_cells(avail_cells: list[tuple[str, list[tuple[int, int]]]]) -> list[tuple[str, int, int]]:
    """把按图片分组的候选拍平成一维列表 [(stem, x, y), ...],方便打乱后顺序消费。"""
    return [(stem, x, y) for stem, cells in avail_cells for (x, y) in cells]

def check_patch(patch: np.ndarray) -> float:
    """整块像素方差。纯色/天空这种低纹理区域方差会很低。"""
    return float(np.var(patch))


def detect_blank_block(patch: np.ndarray, grid: int = BLANK_GRID,
                        flat_var_thresh: float = FLAT_VAR_THRESH, min_frac: float = BLANK_MIN_FRAC,
                        max_border_touch: int = 1) -> bool:
    """检测 cell 内部是否存在一片"孤岛式"的无纹理区域(打码块的典型特征:
    内部平滑无边缘,四周被正常纹理包围,和真实喷漆的平滑面不是一回事)。

    整块方差(check_patch)抓不住局部伪影——block 局部有伪影、局部是正常纹理时,
    两者混在一起整体方差不低不高,骗过纯方差阈值。这里先切网格,各自算局部方差,
    找"局部方差精确趋近 0 的小格子连成一片"的情况,不看整体方差。

    (最初用 Canny 边缘密度做同样的事,但暗光图整体边缘就稀疏,固定边缘阈值会把
    正常的低对比度纹理也判成打码;局部方差对曝光水平不敏感,自然图像的传感器噪声
    /量化几乎不可能让方差精确为 0,只有数字插入的纯色块会这样。)

    光有"平滑区域占比"这一条还不够——集装箱本身的纯色喷漆(比如波纹板凹槽处)
    局部方差也会很低,但这种平滑区域是连续延伸出画面外的物理表面,裁到 cell 里
    通常会贴住多条边界;打码矩形是完整嵌在正常纹理里的"孤岛",一般只贴住 0~1 条边。
    所以额外要求:平滑连通区域贴住的 cell 边界数 <= max_border_touch 才算数。
    (整个 cell 几乎全平的情况,比如天空、大片打码,不靠这条也会被 check_patch 的
    整体方差阈值挡掉,不需要这里再兜底。)
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    step_h, step_w = h // grid, w // grid

    flat_mask = np.zeros((grid, grid), dtype=np.uint8)
    for i in range(grid):
        for j in range(grid):
            block = gray[i * step_h:(i + 1) * step_h, j * step_w:(j + 1) * step_w]
            if block.var() < flat_var_thresh:
                flat_mask[i, j] = 1

    n_labels, labels = cv2.connectedComponents(flat_mask, connectivity=4)
    total_cells = grid * grid
    for label_id in range(1, n_labels):  # 0 是背景(非平滑格子),跳过
        mask = labels == label_id
        if mask.sum() / total_cells < min_frac:
            continue
        rows = np.any(mask, axis=1).nonzero()[0]
        cols = np.any(mask, axis=0).nonzero()[0]
        touch = 0
        if rows.min() == 0:
            touch += 1
        if rows.max() == grid - 1:
            touch += 1
        if cols.min() == 0:
            touch += 1
        if cols.max() == grid - 1:
            touch += 1
        if touch <= max_border_touch:
            return True
    return False


def is_cell_valid(patch: np.ndarray) -> bool:
    """综合两层检测:整体太平(方差低)或局部有打码块,都判定为不可用。"""
    if check_patch(patch) < VAR_THRESH:
        return False
    if detect_blank_block(patch):
        return False
    return True

def stitch_one(patches: list[tuple[str, int, int]]) -> np.ndarray:
    """给定 4 个 (stem, x, y),从对应原图裁出 CELLxCELL 块,拼成一张 CANVASxCANVAS 图。
    patches 长度必须是 4(2x2 网格)。
    """
    canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    positions = [(0, 0), (0, CELL), (CELL, 0), (CELL, CELL)]
    for (stem, x, y), (py, px) in zip(patches, positions):
        img = cv2.imread(str(IMG_PATH / (stem + ".jpg")))
        patch = img[y:y + CELL, x:x + CELL]
        canvas[py:py + CELL, px:px + CELL] = patch
    return canvas


def make_imgs(avail_cells: list[tuple[str, list[tuple[int, int]]]]) -> None:
    """批量生成 TOTAL 张合成负样本,前半存 tune,后半存 eval,互不重复。"""
    flat = flatten_cells(avail_cells)
    random.seed(SEED)
    random.shuffle(flat)

    need = TOTAL * 4
    if len(flat) < need:
        print(f"警告:候选池只有 {len(flat)} 个,不够 {need} 个(={TOTAL}x4),"
              f"会少生成 {TOTAL - len(flat) // 4} 张")

    (OUT_DIR / "tune").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval").mkdir(parents=True, exist_ok=True)

    idx = 0
    for i in range(TOTAL):
        batch = flat[idx:idx + 4]
        if len(batch) < 4:
            break
        idx += 4
        canvas = stitch_one(batch)
        split = "tune" if i < TOTAL // 2 else "eval"
        out_path = OUT_DIR / split / f"syn_neg_{i:04d}.jpg"
        cv2.imwrite(str(out_path), canvas)

    print(f"生成完成,共 {min(TOTAL, idx // 4)} 张,存至 {OUT_DIR}")


if __name__ == "__main__":
    cells = get_all_cells()
    print(f"扫描完成,{len(cells)} 张图有可用干净块")
    make_imgs(cells)

