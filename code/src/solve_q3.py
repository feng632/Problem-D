# -*- coding: utf-8 -*-
"""问题三(第一批,零训练成本):鲁棒性评估 + top-4 排序对比 + NMS/分辨率灵敏度。

===== 范围说明 =====
Q3 checklist 里这几项都只需要对 solve_q2.py 训练好的 best.pt 做多种条件下的
**推理**,不需要重新训练/微调模型权重:
  - 扰动鲁棒性:5 种光度扰动(不改变几何,GT 框坐标不用跟着变)下的 mAP@0.5、F1、
    衰减率 Δj
  - top-4 排序对比:同一批预测框,分别按面积/置信度截断到每图 <=4 框,比较 mAP@0.5
  - NMS 阈值(τ_nms)灵敏度扫描
  - 输入分辨率(640/960/1280)灵敏度扫描(mAP@0.5 与 FPS 双轴)
YOLOv8n/m-seg 对比、消融实验(都要整跑 YOLO 训练)、Grad-CAM(需要挂钩子到模型
中间层,单独实现)不在本脚本范围内。

===== 复用说明 =====
AP 计算(_compute_ap_for_class / _box_iou_matrix / _ap_from_pr)、top-4 截断
(nms_and_topk)、数据集划分(build_seg_dataset)直接从 solve_q2.py 导入,
避免重复实现/口径不一致。评估循环本身重写了一份(evaluate_boxes),因为这里
需要在喂进模型前对图像数组做扰动、且要支持传入 imgsz/iou 覆盖 solve_q2 里
写死的模块常量,和 solve_q2.evaluate_official 的签名对不上。

===== F1 的口径 =====
检测框级别的 P/R/F1(不是 q1 的图像级二分类 F1):在某个置信度阈值下,
按类别、按图做 IoU>=0.5 贪心匹配数 TP/FP/FN,跨三类求和后算 micro F1。
conf* 只在 baseline(未扰动)数据上选一次(对齐 q1 的"调参/评估不混用"原则的精神:
比较"同一个操作点在扰动前后的衰减",而不是"每种扰动各自选最优操作点"——
后者会把模型的自适应能力也算进鲁棒性里,失去了衡量鲁棒性的意义)。
"""
import argparse
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import joblib
import numpy as np
import torch

from solve_q2 import (
    BATCH,
    DATA_DIR,
    EVAL_CHUNK,
    IMG_SIZE,
    IOU_NMS_EVAL,
    NUM_CLASSES,
    RESULT_PKL as Q2_RESULT_PKL,
    SEED,
    SEG_DATASET,
    SRC_DATASET,
    WORKERS,
    _box_iou_matrix,
    _compute_ap_for_class,
    build_seg_dataset,
    nms_and_topk,
)

RESULT_PKL = DATA_DIR / "q3_result.pkl"
TAU_GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)
NMS_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
RES_GRID = [640, 960, 1280]


# ==========================================================================
# 1. 扰动函数(仅像素级光度变换,不改变几何,GT 框坐标沿用原图坐标)
# ==========================================================================
def _brightness(img, factor):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _contrast(img, factor):
    return np.clip((img.astype(np.float32) - 128) * factor + 128, 0, 255).astype(np.uint8)


def _gauss_noise(img, rng, sigma=15.0):
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _gauss_blur(img, ksize=7):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


PERTURBATIONS = {
    "亮度x0.7": lambda im, rng: _brightness(im, 0.7),
    "亮度x1.3": lambda im, rng: _brightness(im, 1.3),
    "对比度x1.3": lambda im, rng: _contrast(im, 1.3),
    "高斯噪声": lambda im, rng: _gauss_noise(im, rng, 15.0),
    "高斯模糊": lambda im, rng: _gauss_blur(im, 7),
}


# ==========================================================================
# 2. 评估循环:给定 imgsz / iou_nms / 扰动函数,跑一遍推理,收集预测与真值
# ==========================================================================
def evaluate_boxes(model, test_stems, imgsz=IMG_SIZE, iou_nms=IOU_NMS_EVAL, conf=0.001, perturb_fn=None, rng=None):
    preds_by_cls = {c: [] for c in range(NUM_CLASSES)}
    gts_by_cls = {c: {} for c in range(NUM_CLASSES)}
    per_image = {}  # img_id -> [{cls, score, box}, ...],供 top-4 实验复用

    t0 = time.time()
    n_imgs = 0
    for chunk_start in range(0, len(test_stems), EVAL_CHUNK):
        chunk_stems = test_stems[chunk_start:chunk_start + EVAL_CHUNK]
        chunk_imgs = []
        for stem in chunk_stems:
            img = cv2.imread(str(SEG_DATASET / "images" / "test" / f"{stem}.jpg"))
            if perturb_fn is not None:
                img = perturb_fn(img, rng)
            chunk_imgs.append(img)

        chunk_results = model.predict(
            source=chunk_imgs,
            imgsz=imgsz,
            batch=BATCH,
            workers=WORKERS,
            conf=conf,  # 默认 0.001(全框口径,后续按需在 conf_thresh 上二次筛选);
                        # 分辨率灵敏度里额外传入 conf_star 单独测"部署口径"下的 FPS
            iou=iou_nms,
            verbose=False,
            retina_masks=False,  # 只需要框,不需要掩码
            save=False,
        )
        for local_id, (stem, res) in enumerate(zip(chunk_stems, chunk_results)):
            img_id = chunk_start + local_id
            h, w = res.orig_shape
            gt_txt = SRC_DATASET / "labels" / "test" / f"{stem}.txt"
            if gt_txt.exists() and gt_txt.read_text().strip():
                for line in gt_txt.read_text().strip().splitlines():
                    c, xc, yc, bw, bh = line.split()
                    c = int(c)
                    xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
                    x1, y1, x2, y2 = (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h
                    area = max(x2 - x1, 0) * max(y2 - y1, 0)
                    gts_by_cls[c].setdefault(img_id, []).append(
                        {"box": [x1, y1, x2, y2], "area": area, "used": False}
                    )

            n_pred = 0 if res.boxes is None else len(res.boxes)
            img_boxes = []
            if n_pred > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                for k in range(n_pred):
                    box = xyxy[k].tolist()
                    area = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
                    c = int(clss[k])
                    sc = float(confs[k])
                    preds_by_cls[c].append((img_id, sc, box, area))
                    img_boxes.append({"cls": c, "score": sc, "box": box})
            per_image[img_id] = img_boxes
            n_imgs += 1
        del chunk_results
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    return preds_by_cls, gts_by_cls, per_image, elapsed, n_imgs


def mean_ap50(preds_by_cls, gts_by_cls):
    aps = []
    for c in range(NUM_CLASSES):
        ap, _, _ = _compute_ap_for_class(preds_by_cls[c], gts_by_cls[c], 0.5)
        aps.append(ap)
    return float(np.mean(aps)), aps


def detection_f1_at_conf(preds_by_cls, gts_by_cls, conf_thresh, iou_thres=0.5):
    """框级别 micro P/R/F1(跨三类求和 TP/FP/FN),与 q1 的图像级二分类 F1 是两回事。"""
    tp_total = fp_total = fn_total = 0
    for c in range(NUM_CLASSES):
        gts_c = gts_by_cls[c]
        for glist in gts_c.values():
            for g in glist:
                g["used"] = False
        n_gt = sum(len(v) for v in gts_c.values())
        preds_c = sorted((p for p in preds_by_cls[c] if p[1] >= conf_thresh), key=lambda x: -x[1])
        tp = fp = 0
        for img_id, score, box, area in preds_c:
            glist = gts_c.get(img_id, [])
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(glist):
                if g["used"]:
                    continue
                iou = _box_iou_matrix([box], [g["box"]])[0, 0]
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thres and best_j >= 0:
                glist[best_j]["used"] = True
                tp += 1
            else:
                fp += 1
        fn = n_gt - tp
        tp_total += tp
        fp_total += fp
        fn_total += fn
    precision = tp_total / max(tp_total + fp_total, 1e-9)
    recall = tp_total / max(tp_total + fn_total, 1e-9)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def evaluate_top4(per_image, gts_by_cls, sort_by):
    """对同一批预测框做类别感知 NMS + top-4 截断(sort_by="area"/"score"),
    用截断后的框重新算 per-class AP@0.5 再取均值。"""
    preds_by_cls = {c: [] for c in range(NUM_CLASSES)}
    for img_id, boxes in per_image.items():
        if not boxes:
            continue
        xyxy = np.array([b["box"] for b in boxes])
        scores = np.array([b["score"] for b in boxes])
        classes = np.array([b["cls"] for b in boxes])
        keep = nms_and_topk(xyxy, scores, classes, iou_thres=IOU_NMS_EVAL, topk=4, sort_by=sort_by)
        for i in keep:
            b = boxes[int(i)]
            box = b["box"]
            area = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
            preds_by_cls[b["cls"]].append((img_id, b["score"], box, area))
    return mean_ap50(preds_by_cls, gts_by_cls)


# ==========================================================================
# 3. 主流程
# ==========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="冒烟测试:官方验证集只取前 10 张")
    args = parser.parse_args()

    if not Q2_RESULT_PKL.exists():
        print(f"[solve_q3] 找不到 {Q2_RESULT_PKL},请先跑 solve_q2.py 训练检测器")
        sys.exit(1)
    q2_result = joblib.load(Q2_RESULT_PKL)
    best_pt = q2_result["best_weights"]
    print(f"[solve_q3] mode={'SMOKE' if args.smoke else 'FULL'}  复用 q2 检测器权重(不训练) -> {best_pt}")

    _, _, _, test_stems = build_seg_dataset(smoke=args.smoke)
    print(f"[solve_q3] 官方验证集 N={len(test_stems)}")

    from ultralytics import YOLO

    model = YOLO(str(best_pt))
    rng = np.random.default_rng(SEED)

    # ---- baseline(clean, imgsz=IMG_SIZE, iou=IOU_NMS_EVAL)----
    print("[solve_q3] baseline(clean) 推理 ...")
    preds_clean, gts_clean, per_image_clean, _, _ = evaluate_boxes(model, test_stems)
    map50_clean, _ = mean_ap50(preds_clean, gts_clean)
    f1_grid = [detection_f1_at_conf(preds_clean, gts_clean, float(t))[2] for t in TAU_GRID]
    conf_star = float(TAU_GRID[int(np.argmax(f1_grid))])
    _, _, f1_clean = detection_f1_at_conf(preds_clean, gts_clean, conf_star)
    print(f"[solve_q3] baseline mAP@0.5={map50_clean:.4f}  conf*={conf_star:.2f}(框级 F1 最优)  F1={f1_clean:.4f}")

    # ---- 鲁棒性:5 种扰动,固定用 baseline 选出的 conf* 计算 F1 ----
    robust_rows = [
        {"perturb": "clean(基准)", "mAP50": round(map50_clean, 4), "F1": round(f1_clean, 4),
         "delta_map_pct": 0.0, "delta_f1_pct": 0.0}
    ]
    for name, fn in PERTURBATIONS.items():
        print(f"[solve_q3] 扰动:{name} 推理 ...")
        preds_p, gts_p, _, _, _ = evaluate_boxes(model, test_stems, perturb_fn=fn, rng=rng)
        map50_p, _ = mean_ap50(preds_p, gts_p)
        _, _, f1_p = detection_f1_at_conf(preds_p, gts_p, conf_star)
        delta_map = (map50_clean - map50_p) / max(map50_clean, 1e-9) * 100
        delta_f1 = (f1_clean - f1_p) / max(f1_clean, 1e-9) * 100
        robust_rows.append(
            {"perturb": name, "mAP50": round(map50_p, 4), "F1": round(f1_p, 4),
             "delta_map_pct": round(delta_map, 2), "delta_f1_pct": round(delta_f1, 2)}
        )
        print(f"    mAP@0.5={map50_p:.4f}(Δ={delta_map:+.2f}%)  F1={f1_p:.4f}(Δ={delta_f1:+.2f}%)")

    # ---- NMS 阈值灵敏度 ----
    nms_curve = []
    for iou_t in NMS_GRID:
        print(f"[solve_q3] NMS iou_nms={iou_t} 推理 ...")
        preds_n, gts_n, _, _, _ = evaluate_boxes(model, test_stems, iou_nms=iou_t)
        map50_n, _ = mean_ap50(preds_n, gts_n)
        nms_curve.append({"iou_nms": iou_t, "mAP50": round(map50_n, 4)})
        print(f"    mAP@0.5={map50_n:.4f}")

    # ---- 输入分辨率灵敏度(mAP@0.5 与 FPS)----
    # FPS 测两档:
    #   fps_eval001: conf=0.001(跟 mAP 用同一次推理算出来的),这是"全召回评估口径"
    #     下的速度,不是真实部署速度——conf 越低候选框越多,NMS/后处理成本会被
    #     不成比例地放大(尤其在 P2 头 + 高分辨率下,实测 1280 分辨率能有 ~93 个
    #     候选框/图,vs conf=0.20 时只有 ~2 个/图)。
    #   fps_deploy: 额外单独用 conf=conf_star(baseline 上选出的部署阈值)重新推理
    #     一遍只为计时,这个数字才是"实际部署会看到的速度",画图/写论文用这个。
    res_curve = []
    for sz in RES_GRID:
        print(f"[solve_q3] 分辨率 imgsz={sz} 推理(mAP 口径 conf=0.001)...")
        preds_r, gts_r, _, elapsed_r, n_r = evaluate_boxes(model, test_stems, imgsz=sz)
        map50_r, _ = mean_ap50(preds_r, gts_r)
        fps_eval001 = n_r / elapsed_r if elapsed_r > 0 else 0.0

        print(f"[solve_q3] 分辨率 imgsz={sz} 计时(部署口径 conf={conf_star:.2f})...")
        _, _, _, elapsed_d, n_d = evaluate_boxes(model, test_stems, imgsz=sz, conf=conf_star)
        fps_deploy = n_d / elapsed_d if elapsed_d > 0 else 0.0

        res_curve.append({
            "imgsz": sz, "mAP50": round(map50_r, 4),
            "fps_eval001": round(fps_eval001, 2), "fps_deploy": round(fps_deploy, 2),
        })
        print(f"    mAP@0.5={map50_r:.4f}  FPS(eval,conf=0.001)={fps_eval001:.2f}"
              f"  FPS(部署,conf={conf_star:.2f})={fps_deploy:.2f}")

    # ---- top-4 排序对比(面积 vs 置信度)----
    print("[solve_q3] top-4 排序对比(面积 vs 置信度) ...")
    map50_area, _ = evaluate_top4(per_image_clean, gts_clean, sort_by="area")
    map50_score, _ = evaluate_top4(per_image_clean, gts_clean, sort_by="score")
    top4_rows = [
        {"sort_by": "area(面积/严重度代理)", "mAP50": round(map50_area, 4)},
        {"sort_by": "score(置信度)", "mAP50": round(map50_score, 4)},
    ]
    print(f"    按面积: mAP@0.5={map50_area:.4f}    按置信度: mAP@0.5={map50_score:.4f}")

    result = {
        "seed": SEED,
        "mode": "smoke" if args.smoke else "full",
        "best_weights": str(best_pt),
        "n_test": len(test_stems),
        "baseline": {"mAP50": round(map50_clean, 4), "conf_star": conf_star, "F1": round(f1_clean, 4)},
        "f1_grid": {"tau_grid": TAU_GRID.tolist(), "f1": [round(float(v), 4) for v in f1_grid]},
        "robustness": robust_rows,
        "nms_sensitivity": nms_curve,
        "resolution_sensitivity": res_curve,
        "top4_compare": top4_rows,
        "note": (
            "全部复用 solve_q2.py 训练好的 best.pt,本脚本不重新训练/微调,只做多种条件下的推理评估。"
            "鲁棒性 F1 用 baseline(未扰动)数据选出的单一 conf* 计算,扰动前后共用同一操作点,"
            "不为每种扰动重新调阈值(否则测的是模型自适应能力,不是鲁棒性)。"
            "FPS 计时包含图像读取与扰动/预处理开销,不是纯 GPU 前向计时。"
            "resolution_sensitivity 里 fps_eval001 是 mAP 计算用的 conf=0.001 口径下的速度"
            "(候选框多,NMS/后处理被放大,不代表真实部署速度,只留作记录);"
            "fps_deploy 是额外用 conf_star(部署阈值)单独计时得到的,写论文/画图用这个。"
        ),
    }
    joblib.dump(result, RESULT_PKL)
    print(f"[solve_q3] 结果已保存 -> {RESULT_PKL}")
    print("[solve_q3] 完成(本批:鲁棒性+top-4+NMS/分辨率灵敏度)。"
          "YOLOv8n/m-seg 对比、消融实验、Grad-CAM 未包含在本脚本内。")


if __name__ == "__main__":
    main()
