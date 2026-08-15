"""演示脚本:合成负样本拼图,给用户看一眼效果,不是正式 solve_q1.py 的一部分。"""
import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import cv2

DATA = Path(__file__).resolve().parent.parent / "data" / "dataset_3713"
IMG_DIR = DATA / "images" / "train"
LBL_DIR = DATA / "labels" / "train"

CELL = 320  # 2x2 网格,每格 320x320,拼成 640x640
MARGIN = 12  # 标注框外扩边距(像素),避开框边缘的残损


def load_boxes_px(label_path, w, h):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        _, xc, yc, bw, bh = map(float, line.split())
        x1 = (xc - bw / 2) * w - MARGIN
        y1 = (yc - bh / 2) * h - MARGIN
        x2 = (xc + bw / 2) * w + MARGIN
        y2 = (yc + bh / 2) * h + MARGIN
        boxes.append((x1, y1, x2, y2))
    return boxes


def cell_is_free(cx1, cy1, cx2, cy2, boxes):
    """网格 cell 是否与任何(外扩后的)标注框有重叠"""
    for (x1, y1, x2, y2) in boxes:
        if not (cx2 <= x1 or cx1 >= x2 or cy2 <= y1 or cy1 >= y2):
            return False
    return True


def find_free_cells(img_path, label_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    boxes = load_boxes_px(label_path, w, h)
    free = []
    # 用比 CELL 稍密的网格扫描,增加候选数量
    step = CELL // 2
    for y in range(0, h - CELL + 1, step):
        for x in range(0, w - CELL + 1, step):
            if cell_is_free(x, y, x + CELL, y + CELL, boxes):
                free.append(img[y:y + CELL, x:x + CELL].copy())
    return free


def main():
    random.seed(42)
    all_imgs = sorted(IMG_DIR.glob("*.jpg"))
    random.shuffle(all_imgs)

    patches = []
    tried = 0
    for img_path in all_imgs:
        tried += 1
        label_path = LBL_DIR / (img_path.stem + ".txt")
        free = find_free_cells(img_path, label_path)
        if free:
            patches.append((img_path.name, random.choice(free)))
        if len(patches) >= 4 or tried > 200:
            break

    print(f"扫描了 {tried} 张图,找到 {len(patches)} 个可用 320x320 空白块,来源:")
    for name, _ in patches:
        print(" -", name)

    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    positions = [(0, 0), (0, CELL), (CELL, 0), (CELL, CELL)]
    for (name, patch), (py, px) in zip(patches, positions):
        canvas[py:py + CELL, px:px + CELL] = patch
        cv2.rectangle(canvas, (px, py), (px + CELL - 1, py + CELL - 1), (0, 255, 0), 2)
        cv2.putText(canvas, name, (px + 5, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out = Path(__file__).resolve().parent.parent / "figures" / "demo_syn_neg.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print("saved:", out)


if __name__ == "__main__":
    main()
