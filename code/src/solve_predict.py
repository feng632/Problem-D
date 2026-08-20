# -*- coding: utf-8 -*-
"""submission/test_result.csv —— 演示提交文件(413 张官方验证集)。
复用 q2 训练好的 best.pt(不训练),对测试集跑推理,类别感知 NMS +
按面积(严重度代理)截断到每图 <=4 框,输出归一化 [image_id, class_id,
x_center, y_center, width, height]。

原赛题除论文外还要交 test_result.csv;作业语境下主交付物是论文,这里
用带标注的 413 张验证集生成"演示版",证明模型端到端可用——若老师后续
发布真实无标注测试集,把 SEG_DATASET/images/test 换成新目录重跑即可,
其余逻辑不用改(推理不依赖标注,GT 标注文件本脚本完全不读)。
image_id 用图像文件名(不含扩展名),不是内部枚举下标,方便对着原图核对。
"""
import csv
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import joblib

from solve_q2 import (
    BATCH,
    EVAL_CHUNK,
    IMG_SIZE,
    IOU_NMS_EVAL,
    RESULT_PKL as Q2_RESULT_PKL,
    SEG_DATASET,
    WORKERS,
    build_seg_dataset,
    nms_and_topk,
)

ROOT = Path(__file__).resolve().parents[1]
SUBM_DIR = ROOT.parent / "submission"
OUT_CSV = SUBM_DIR / "test_result.csv"


def main():
    if not Q2_RESULT_PKL.exists():
        print(f"[solve_predict] 找不到 {Q2_RESULT_PKL},请先跑 solve_q2.py 训练检测器")
        sys.exit(1)
    q2_result = joblib.load(Q2_RESULT_PKL)
    best_pt = q2_result["best_weights"]
    print(f"[solve_predict] 复用 q2 检测器权重(不训练) -> {best_pt}")

    _, _, _, test_stems = build_seg_dataset()
    print(f"[solve_predict] 官方验证集 N={len(test_stems)}(演示提交,老师发布真实测试集后换目录重跑)")

    from ultralytics import YOLO

    model = YOLO(str(best_pt))

    rows = []
    for chunk_start in range(0, len(test_stems), EVAL_CHUNK):
        chunk_stems = test_stems[chunk_start:chunk_start + EVAL_CHUNK]
        chunk_imgs = [cv2.imread(str(SEG_DATASET / "images" / "test" / f"{s}.jpg")) for s in chunk_stems]
        chunk_results = model.predict(
            source=chunk_imgs,
            imgsz=IMG_SIZE,
            batch=BATCH,
            workers=WORKERS,
            conf=0.001,
            iou=IOU_NMS_EVAL,
            verbose=False,
            retina_masks=False,
            save=False,
        )
        for stem, res in zip(chunk_stems, chunk_results):
            h, w = res.orig_shape
            n_pred = 0 if res.boxes is None else len(res.boxes)
            if n_pred == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            keep = nms_and_topk(xyxy, confs, clss, iou_thres=IOU_NMS_EVAL, topk=4, sort_by="area")
            for i in keep:
                x1, y1, x2, y2 = xyxy[int(i)]
                c = int(clss[int(i)])
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                rows.append([stem, c, round(float(xc), 6), round(float(yc), 6), round(float(bw), 6), round(float(bh), 6)])
        print(f"[solve_predict] {chunk_start + len(chunk_stems)}/{len(test_stems)}")

    SUBM_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "class_id", "x_center", "y_center", "width", "height"])
        writer.writerows(rows)
    print(f"[solve_predict] -> {OUT_CSV}  共 {len(rows)} 行(<=4/图 x {len(test_stems)} 张)")


if __name__ == "__main__":
    main()
