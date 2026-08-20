"""演示脚本:可视化"干净块"扫描过程——画出标注框(原始+外扩)与网格扫描结果。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import cv2

DATA = Path(__file__).resolve().parent.parent / "data" / "dataset_3713"
IMG_DIR = DATA / "images" / "train"
LBL_DIR = DATA / "labels" / "train"

CELL = 320
MARGIN = 12
STEP = CELL // 2


def load_boxes_px(label_path, w, h):
    raw, expanded = [], []
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        _, xc, yc, bw, bh = map(float, line.split())
        x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
        x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
        raw.append((x1, y1, x2, y2))
        expanded.append((x1 - MARGIN, y1 - MARGIN, x2 + MARGIN, y2 + MARGIN))
    return raw, expanded


def cell_is_free(cx1, cy1, cx2, cy2, boxes):
    for (x1, y1, x2, y2) in boxes:
        if not (cx2 <= x1 or cx1 >= x2 or cy2 <= y1 or cy1 >= y2):
            return False
    return True


def main(stem="1"):
    img_path = IMG_DIR / f"{stem}.jpg"
    label_path = LBL_DIR / f"{stem}.txt"
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    raw_boxes, exp_boxes = load_boxes_px(label_path, w, h)

    vis = img.copy()
    # 原始标注框:红色实线
    for (x1, y1, x2, y2) in raw_boxes:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    # 外扩后的排除区:黄色虚线(用短线段模拟虚线)
    for (x1, y1, x2, y2) in exp_boxes:
        pts = [(int(x1), int(y1)), (int(x2), int(y1)), (int(x2), int(y2)), (int(x1), int(y2))]
        for i in range(4):
            p1, p2 = pts[i], pts[(i + 1) % 4]
            n = 12
            for t in range(0, n, 2):
                a = (p1[0] + (p2[0] - p1[0]) * t / n, p1[1] + (p2[1] - p1[1]) * t / n)
                b = (p1[0] + (p2[0] - p1[0]) * (t + 1) / n, p1[1] + (p2[1] - p1[1]) * (t + 1) / n)
                cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 255), 2)

    # 先描出所有被扫描到的格子边框(细线,不上色),再单独把"干净块"用绿色半透明填充,避免重叠格子的颜色互相叠加糊成一片
    cells = []
    for y in range(0, h - CELL + 1, STEP):
        for x in range(0, w - CELL + 1, STEP):
            cells.append((x, y, cell_is_free(x, y, x + CELL, y + CELL, exp_boxes)))

    for (x, y, free) in cells:
        cv2.rectangle(vis, (x, y), (x + CELL, y + CELL), (150, 150, 150), 1)

    overlay = vis.copy()
    for (x, y, free) in cells:
        if free:
            cv2.rectangle(overlay, (x, y), (x + CELL, y + CELL), (0, 200, 0), -1)
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
    for (x, y, free) in cells:
        if free:
            cv2.rectangle(vis, (x, y), (x + CELL, y + CELL), (0, 220, 0), 2)

    free_count = sum(1 for _, _, f in cells if f)
    occ_count = len(cells) - free_count

    legend_y = 25
    cv2.putText(vis, "red = label box, yellow dashed = expanded exclude zone", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"green cell = candidate clean block ({free_count} found, {occ_count} rejected)",
                (10, legend_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out = Path(__file__).resolve().parent.parent / "figures" / "demo_free_cell_scan.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), vis)
    print(f"sample={stem}  boxes={len(raw_boxes)}  free_cells={free_count}  occupied_cells={occ_count}")
    print("saved:", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1")
