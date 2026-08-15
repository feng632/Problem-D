# -*- coding: utf-8 -*-
"""问题二:集装箱残损检测/实例分割(ultralytics YOLOv8-seg)。

产出(保存到 code/data/q2_result.pkl,供 fig_q2_*.py 读取出图):
  - 训练损失收敛曲线数据(来自 ultralytics runs/.../results.csv)
  - 官方验证集(413 张)上:per-class AP@0.5、AP@0.5:0.95、AP_s/AP_m/AP_l、
    per-class Dice、mIoU、PR 曲线数据、混淆矩阵(3 类 + 背景)
  - 4 张随机可视化样例的原始预测(框 + 掩码 + 类别),供 fig_q2_visual.py 画图
  - 训练好的权重路径

===== 重要说明(数据集实际情况,与题面假设不同,已按此调整实现)=====
`code/data/dataset_3713/labels/*.txt` 是纯 YOLO **检测**格式
(`class xc yc w h`,5 个字段),**没有**逐像素分割多边形标注。
为了仍能训练/评估 YOLOv8-**seg**,本脚本把每个 bbox 当作其自身的矩形
"伪分割掩码"(四个角点的多边形),写入 `code/data/dataset_3713_seg/labels/`。
这意味着:
  1) 训练出的分割头学习的是"贴合矩形框"的掩码,不是真实残损轮廓;
  2) 报告的 Dice / mIoU 是"预测掩码 vs. 矩形伪真值"的一致性指标,
     不是与人工像素级标注的对比,解读时需注明这一点(已在 stdout 汇总与
     figures 标题中提示)。
这是在给定数据集约束下的工程折中,而非擅自更改题目要求。

===== 类别加权说明 =====
ultralytics 8.4.120 的 `cls_pw` 超参(0~1)原生支持"逆类别频率加权"
(1.0 = 完全按逆频率加权,0.0 = 关闭),训练/分割 Trainer 都实现了
`set_class_weights()`。本脚本直接使用 `cls_pw=1.0`,无需魔改内部 loss。

===== 训练/验证划分 =====
从 train 3300 张中以固定随机种子 SEED=42 划出 10%(330 张)作为
"训练期验证集"(早停 / 选模型);官方 413 张 test 集只在训练完全结束后
调用一次 `model.predict` 做最终评估,不参与训练期的模型选择,避免选择泄漏。
"""
import argparse
import json
import random
import subprocess
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
import torchvision

# --------------------------------------------------------------------------
# 配置(路径 / 超参),真实训练把 EPOCHS 调大即可,不用改别的
# --------------------------------------------------------------------------
SEED = 42
IMG_SIZE = 1280
BATCH = 4  # 8GB 显存 + 1280 分辨率 + P2 头 + seg,保守取值
EPOCHS = 100  # 完整训练轮数
SMOKE_EPOCHS = 2  # 冒烟测试轮数
PATIENCE = 20  # 早停 patience(基于训练期验证集)
BASE_WEIGHTS = "yolov8s-seg.pt"  # 起步权重(兼顾速度/精度)
CONF_EVAL = 0.001  # 评估阶段低置信度阈值,保证全框口径召回充分
IOU_NMS_EVAL = 0.6

ROOT = Path(__file__).resolve().parents[1]  # code/
SRC_DIR = Path(__file__).resolve().parent
MODEL_CFG = SRC_DIR / "yolov8s-seg-p2.yaml"
DATA_DIR = ROOT / "data"
SRC_DATASET = DATA_DIR / "dataset_3713"
SEG_DATASET = DATA_DIR / "dataset_3713_seg"
RUNS_DIR = DATA_DIR / "runs_q2"
RESULT_PKL = DATA_DIR / "q2_result.pkl"

CLASS_NAMES = {0: "Dent", 1: "Hole", 2: "Rusty"}
NUM_CLASSES = 3

AREA_SMALL = 32 * 32
AREA_LARGE = 96 * 96


# ==========================================================================
# 1. 数据集准备:bbox -> 矩形伪分割标签,90/10 训练期划分,写 data.yaml
# ==========================================================================
def _make_junction(link: Path, target: Path):
    """Windows 目录联接(不复制文件,不需要管理员权限)。已存在则跳过。"""
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f'New-Item -ItemType Junction -Path "{link}" -Target "{target}" | Out-Null'
    )
    subprocess.run(
        ["powershell", "-NonInteractive", "-NoProfile", "-Command", cmd],
        check=True,
    )


def _bbox_to_poly_label(src_txt: Path, dst_txt: Path):
    if dst_txt.exists():
        return
    lines_out = []
    text = src_txt.read_text().strip()
    if text:
        for line in text.splitlines():
            if not line.strip():
                continue
            parts = line.split()
            c = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
            x1, x2 = float(np.clip(x1, 0, 1)), float(np.clip(x2, 0, 1))
            y1, y2 = float(np.clip(y1, 0, 1)), float(np.clip(y2, 0, 1))
            lines_out.append(
                f"{c} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"
            )
    dst_txt.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))


def class_distribution(label_dir: Path):
    counts = {c: 0 for c in range(NUM_CLASSES)}
    for f in label_dir.glob("*.txt"):
        for line in f.read_text().strip().splitlines():
            if not line.strip():
                continue
            c = int(line.split()[0])
            counts[c] = counts.get(c, 0) + 1
    return counts


def build_seg_dataset(smoke: bool = False):
    """返回 (data_yaml_path, train_sub_stems, val_sub_stems, test_stems_used)。"""
    (SEG_DATASET / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (SEG_DATASET / "labels" / "test").mkdir(parents=True, exist_ok=True)
    (SEG_DATASET / "images").mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        _make_junction(SEG_DATASET / "images" / split, SRC_DATASET / "images" / split)
        src_lbl_dir = SRC_DATASET / "labels" / split
        dst_lbl_dir = SEG_DATASET / "labels" / split
        for src in src_lbl_dir.glob("*.txt"):
            _bbox_to_poly_label(src, dst_lbl_dir / src.name)

    rng = random.Random(SEED)  # 固定随机种子 SEED=42,划分可复现
    train_stems = sorted(p.stem for p in (SRC_DATASET / "images" / "train").glob("*.jpg"))
    rng.shuffle(train_stems)
    n_val = round(len(train_stems) * 0.10)
    val_stems = sorted(train_stems[:n_val])
    train_sub_stems = sorted(train_stems[n_val:])
    test_stems = sorted(p.stem for p in (SRC_DATASET / "images" / "test").glob("*.jpg"))

    if smoke:
        train_sub_stems = train_sub_stems[:40]
        val_stems = val_stems[:10]
        test_stems_used = test_stems[:10]
    else:
        test_stems_used = test_stems

    def _write_list(name, stems, split_dir):
        # 注意:不能用 Path.resolve(),它会把 Windows 目录联接(junction)解析回
        # 物理目标路径(dataset_3713/images/...),导致 ultralytics 按
        # "images"->"labels" 字符串替换时又跑回原始 bbox 格式标签目录。
        # 用 os.path.abspath 只做语法规整,不追踪 reparse point。
        import os as _os

        p = SEG_DATASET / name
        img_dir = SEG_DATASET / "images" / split_dir
        p.write_text("\n".join(_os.path.abspath(str(img_dir / f"{s}.jpg")) for s in stems) + "\n")
        return p

    train_list = _write_list("split_train.txt", train_sub_stems, "train")
    val_list = _write_list("split_valsub.txt", val_stems, "train")
    test_list = _write_list("split_test.txt", test_stems_used, "test")

    data_yaml = SEG_DATASET / "data.yaml"
    yaml_text = (
        f"path: {SEG_DATASET.resolve()}\n"
        f"train: {train_list.name}\n"
        f"val: {val_list.name}\n"
        f"test: {test_list.name}\n"
        f"names:\n  0: Dent\n  1: Hole\n  2: Rusty\n"
    )
    data_yaml.write_text(yaml_text)
    return data_yaml, train_sub_stems, val_stems, test_stems_used


# ==========================================================================
# 2. NMS + top-4(按面积)工具函数 —— 供 solve_q3.py 复用
# ==========================================================================
def nms_and_topk(boxes_xyxy, scores, classes, masks=None, iou_thres=0.5, topk=4, sort_by="area"):
    """类别感知 NMS,再按 sort_by 排序截断到每图最多 topk 个框。

    Args:
        boxes_xyxy: (N,4) array-like,像素坐标 [x1,y1,x2,y2]
        scores: (N,) 置信度
        classes: (N,) 类别 id
        masks: 可选 (N,H,W) bool/0-1 掩码,与 boxes 同步筛选
        iou_thres: NMS IoU 阈值
        topk: 每张图最多保留框数(官方规则:每图 <=4 个框)
        sort_by: "area"(默认,面积作为残损严重度代理)或 "score"

    Returns:
        keep_idx: 原始输入数组下标(int ndarray),长度 <= topk,已按 sort_by 降序排列
    """
    boxes_t = torch.as_tensor(np.asarray(boxes_xyxy), dtype=torch.float32)
    scores_t = torch.as_tensor(np.asarray(scores), dtype=torch.float32)
    classes_t = torch.as_tensor(np.asarray(classes), dtype=torch.float32)
    if boxes_t.numel() == 0:
        return np.array([], dtype=int)
    keep = torchvision.ops.batched_nms(boxes_t, scores_t, classes_t, iou_thres)
    keep = keep.numpy()
    if sort_by == "area":
        b = boxes_t.numpy()[keep]
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        order = np.argsort(-area)
    else:  # "score"
        order = np.argsort(-scores_t.numpy()[keep])
    keep = keep[order][:topk]
    return keep


# ==========================================================================
# 3. 训练
# ==========================================================================
def train_model(data_yaml: Path, epochs: int, run_name: str, batch: int = BATCH):
    from ultralytics import YOLO

    model = YOLO(str(MODEL_CFG))
    model.load(BASE_WEIGHTS)  # 迁移同形状层权重;P2 新增层保持随机初始化

    t0 = time.time()
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=batch,
        seed=SEED,
        deterministic=True,
        patience=PATIENCE,
        cls_pw=1.0,  # 逆类别频率加权(class-weighted loss)
        # mask proto 仍从 P3(stride 8)生成(见 yolov8s-seg-p2.yaml 里的排序说明),
        # 与默认 yolov8-seg 结构一致,mask_ratio 保持默认值 4 即可。
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        plots=True,
        val=True,
        # 强数据增强
        mosaic=1.0,
        mixup=0.10,
        copy_paste=0.10,
        degrees=5.0,
        translate=0.10,
        scale=0.50,
        shear=2.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    elapsed = time.time() - t0
    run_dir = RUNS_DIR / run_name
    best_pt = run_dir / "weights" / "best.pt"
    return model, run_dir, best_pt, elapsed


# ==========================================================================
# 4. 官方验证集(413 张)评估:AP / AP_s,m,l / Dice / mIoU / 混淆矩阵
# ==========================================================================
def _box_iou_matrix(a, b):
    """a:(N,4) b:(M,4) xyxy -> (N,M) IoU"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _ap_from_pr(recall, precision):
    """连续积分(precision envelope)版 AP,近似 COCO 101 点插值。"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def _compute_ap_for_class(preds, gts, iou_thres, area_range=None):
    """preds: list of (img_id, score, box, area) sorted later by score desc.
    gts: dict img_id -> list of dict{box, area, used}
    返回 AP、以及(仅 iou_thres=0.5 时有用的)precision/recall 曲线数组。
    """
    if area_range is not None:
        lo, hi = area_range
        gts_f = {}
        n_gt = 0
        for img_id, glist in gts.items():
            sub = [g for g in glist if lo <= g["area"] < hi]
            gts_f[img_id] = sub
            n_gt += len(sub)
    else:
        gts_f = {k: list(v) for k, v in gts.items()}
        n_gt = sum(len(v) for v in gts_f.values())

    for glist in gts_f.values():
        for g in glist:
            g["used"] = False

    if n_gt == 0:
        return 0.0, np.array([]), np.array([])

    preds_sorted = sorted(preds, key=lambda x: -x[1])
    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))
    for i, (img_id, score, box, area) in enumerate(preds_sorted):
        if area_range is not None and not (lo <= area < hi):
            # 该预测本身不落在目标尺寸桶内:既不算 TP 也不算 FP(COCO 式忽略)
            tp[i] = fp[i] = 0
            continue
        glist = gts_f.get(img_id, [])
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(glist):
            if g["used"]:
                continue
            iou = _box_iou_matrix([box], [g["box"]])[0, 0]
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thres and best_j >= 0:
            glist[best_j]["used"] = True
            tp[i] = 1
        else:
            fp[i] = 1

    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recall = tp_c / max(n_gt, 1)
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    ap = _ap_from_pr(recall, precision)
    return ap, recall, precision


def evaluate_official(model, test_stems, class_names=CLASS_NAMES):
    """在官方 413(或冒烟子集)验证集上跑一次推理并计算全部指标。"""
    img_paths = [str((SEG_DATASET / "images" / "test" / f"{s}.jpg").resolve()) for s in test_stems]
    results = model.predict(
        source=img_paths,
        imgsz=IMG_SIZE,
        conf=CONF_EVAL,
        iou=IOU_NMS_EVAL,
        verbose=False,
        retina_masks=True,
        save=False,
    )

    # ---- 汇总预测 / 真值 ----
    preds_by_cls = {c: [] for c in range(NUM_CLASSES)}  # (img_id, score, box, area)
    gts_by_cls = {c: {} for c in range(NUM_CLASSES)}  # img_id -> [ {box, area, used} ]

    confusion = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=int)  # rows=gt(+bg), cols=pred(+bg)

    viz_samples = []  # 存 4 个随机样例的原始预测,供 fig_q2_visual.py

    rng = random.Random(SEED)
    viz_pick = set(rng.sample(range(len(test_stems)), k=min(4, len(test_stems))))

    for img_id, (stem, res) in enumerate(zip(test_stems, results)):
        h, w = res.orig_shape
        # ground truth (原始 bbox 格式,直接读取,避免多边形反算误差)
        gt_txt = SRC_DATASET / "labels" / "test" / f"{stem}.txt"
        gt_list = []
        if gt_txt.exists() and gt_txt.read_text().strip():
            for line in gt_txt.read_text().strip().splitlines():
                c, xc, yc, bw, bh = line.split()
                c = int(c)
                xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
                x1, y1, x2, y2 = (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h
                area = max(x2 - x1, 0) * max(y2 - y1, 0)
                gt_list.append({"cls": c, "box": [x1, y1, x2, y2], "area": area})
                gts_by_cls[c].setdefault(img_id, []).append({"box": [x1, y1, x2, y2], "area": area, "used": False})

        # predictions
        n_pred = 0 if res.boxes is None else len(res.boxes)
        pred_entries = []  # for this image: (cls, score, box, mask)
        if n_pred > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            if res.masks is not None:
                mdata = res.masks.data.cpu().numpy().astype(bool)  # (n,H,W) already retina (orig size)
            else:
                mdata = None
            for k in range(n_pred):
                box = xyxy[k].tolist()
                area = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
                c = int(clss[k])
                sc = float(confs[k])
                preds_by_cls[c].append((img_id, sc, box, area))
                mask = mdata[k] if mdata is not None else None
                pred_entries.append({"cls": c, "score": sc, "box": box, "mask": mask})

        if img_id in viz_pick:
            viz_samples.append(
                {
                    "stem": stem,
                    "img_path": img_paths[img_id],
                    "orig_shape": (h, w),
                    "gt": gt_list,
                    "pred": [
                        {"cls": p["cls"], "score": p["score"], "box": p["box"]}
                        for p in pred_entries
                    ],
                    "pred_masks": [p["mask"] for p in pred_entries] if pred_entries else [],
                }
            )

        # ---- 混淆矩阵(IoU>=0.5 贪心匹配,3 类 + 背景)----
        gt_boxes_img = [g["box"] for g in gt_list]
        gt_cls_img = [g["cls"] for g in gt_list]
        pred_sorted_idx = sorted(range(len(pred_entries)), key=lambda i: -pred_entries[i]["score"])
        gt_used = [False] * len(gt_list)
        for pi in pred_sorted_idx:
            pbox = pred_entries[pi]["box"]
            pcls = pred_entries[pi]["cls"]
            if gt_boxes_img:
                ious = _box_iou_matrix([pbox], gt_boxes_img)[0]
            else:
                ious = np.array([])
            best_j = -1
            best_iou = 0.5
            for j, iou in enumerate(ious):
                if not gt_used[j] and iou >= best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                gt_used[best_j] = True
                confusion[gt_cls_img[best_j], pcls] += 1
            else:
                confusion[NUM_CLASSES, pcls] += 1  # 误检(背景 -> 类别)
        for j, used in enumerate(gt_used):
            if not used:
                confusion[gt_cls_img[j], NUM_CLASSES] += 1  # 漏检(类别 -> 背景)

    # ---- AP@0.5 / AP@0.5:0.95 / AP_s,m,l (逐类) ----
    iou_range = np.arange(0.5, 1.0, 0.05)
    per_class_metrics = {}
    pr_curves = {}
    for c in range(NUM_CLASSES):
        ap50, rec50, prec50 = _compute_ap_for_class(preds_by_cls[c], gts_by_cls[c], 0.5)
        aps = []
        for thr in iou_range:
            ap_t, _, _ = _compute_ap_for_class(preds_by_cls[c], gts_by_cls[c], float(thr))
            aps.append(ap_t)
        ap5095 = float(np.mean(aps))
        ap_s, _, _ = _compute_ap_for_class(preds_by_cls[c], gts_by_cls[c], 0.5, area_range=(0, AREA_SMALL))
        ap_m, _, _ = _compute_ap_for_class(preds_by_cls[c], gts_by_cls[c], 0.5, area_range=(AREA_SMALL, AREA_LARGE))
        ap_l, _, _ = _compute_ap_for_class(preds_by_cls[c], gts_by_cls[c], 0.5, area_range=(AREA_LARGE, 1e12))
        per_class_metrics[c] = {
            "AP50": round(ap50, 4),
            "AP50_95": round(ap5095, 4),
            "AP_s": round(ap_s, 4),
            "AP_m": round(ap_m, 4),
            "AP_l": round(ap_l, 4),
        }
        pr_curves[c] = {"recall": rec50.round(4).tolist(), "precision": prec50.round(4).tolist()}

    # ---- Dice / mIoU (逐类,GT 实例级平均,贪心按分数匹配) ----
    dice_iou = _compute_dice_miou(results, test_stems, class_names)

    metrics_table = []
    for c in range(NUM_CLASSES):
        row = dict(per_class_metrics[c])
        row["class"] = class_names[c]
        row["Dice"] = dice_iou[c]["dice"]
        row["mIoU"] = dice_iou[c]["iou"]
        metrics_table.append(row)

    return {
        "per_class_metrics": per_class_metrics,
        "pr_curves": pr_curves,
        "dice_iou": dice_iou,
        "confusion": confusion.tolist(),
        "metrics_table": metrics_table,
        "viz_samples": viz_samples,
    }


def _compute_dice_miou(results, test_stems, class_names, iou_match_min=0.1):
    """GT 实例级平均 Dice / IoU(矩形伪掩码 vs 预测掩码);未命中的 GT 记 0。"""
    per_class = {c: [] for c in range(NUM_CLASSES)}
    for img_id, (stem, res) in enumerate(zip(test_stems, results)):
        h, w = res.orig_shape
        gt_txt = SRC_DATASET / "labels" / "test" / f"{stem}.txt"
        gts = []
        if gt_txt.exists() and gt_txt.read_text().strip():
            for line in gt_txt.read_text().strip().splitlines():
                c, xc, yc, bw, bh = line.split()
                c = int(c)
                xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
                x1, y1, x2, y2 = (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h
                m = np.zeros((h, w), dtype=bool)
                m[int(max(y1, 0)):int(min(y2, h)), int(max(x1, 0)):int(min(x2, w))] = True
                gts.append({"cls": c, "mask": m, "used": False})
        if not gts:
            continue

        preds = []
        if res.boxes is not None and len(res.boxes) > 0 and res.masks is not None:
            clss = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            mdata = res.masks.data.cpu().numpy().astype(bool)
            order = np.argsort(-confs)
            for k in order:
                preds.append({"cls": int(clss[k]), "mask": mdata[k]})

        for g in gts:
            best_iou, best_dice, best_p = 0.0, 0.0, None
            for p in preds:
                if p["cls"] != g["cls"] or p.get("used"):
                    continue
                inter = np.logical_and(p["mask"], g["mask"]).sum()
                union = np.logical_or(p["mask"], g["mask"]).sum()
                if union == 0:
                    continue
                iou = inter / union
                dice = 2 * inter / (p["mask"].sum() + g["mask"].sum() + 1e-9)
                if iou > best_iou:
                    best_iou, best_dice, best_p = iou, dice, p
            if best_p is not None and best_iou >= iou_match_min:
                best_p["used"] = True
                per_class[g["cls"]].append((best_dice, best_iou))
            else:
                per_class[g["cls"]].append((0.0, 0.0))

    out = {}
    for c in range(NUM_CLASSES):
        vals = per_class[c]
        if vals:
            dice_mean = float(np.mean([v[0] for v in vals]))
            iou_mean = float(np.mean([v[1] for v in vals]))
        else:
            dice_mean, iou_mean = 0.0, 0.0
        out[c] = {"dice": round(dice_mean, 4), "iou": round(iou_mean, 4)}
    return out


# ==========================================================================
# 5. 训练损失曲线数据(从 ultralytics results.csv 读取)
# ==========================================================================
def load_loss_curve(run_dir: Path):
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    keep_cols = [c for c in df.columns if "loss" in c or c == "epoch"]
    data = {c: df[c].round(4).tolist() for c in keep_cols}
    return data


# ==========================================================================
# 6. 主流程
# ==========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="冒烟测试:小数据子集 + 少量 epoch,验证全流程可跑通")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖默认 epoch 数")
    parser.add_argument("--batch", type=int, default=BATCH)
    args = parser.parse_args()

    epochs = args.epochs if args.epochs is not None else (SMOKE_EPOCHS if args.smoke else EPOCHS)
    run_name = "smoke" if args.smoke else "full"

    print(f"[solve_q2] mode={'SMOKE' if args.smoke else 'FULL'} epochs={epochs} imgsz={IMG_SIZE} batch={args.batch}")

    data_yaml, train_stems, val_stems, test_stems = build_seg_dataset(smoke=args.smoke)
    print(f"[solve_q2] data.yaml -> {data_yaml}")
    print(f"[solve_q2] train_sub={len(train_stems)}  val_sub(training-period)={len(val_stems)}  official_test_used={len(test_stems)}")

    cls_count_train = class_distribution(SRC_DATASET / "labels" / "train")
    total = sum(cls_count_train.values())
    print("[solve_q2] 训练集(3300 张)类别实例分布(class-imbalance 检查):")
    for c in range(NUM_CLASSES):
        n = cls_count_train.get(c, 0)
        print(f"    {CLASS_NAMES[c]:>6s}(id={c}): {n:5d}  ({100*n/total:.2f}%)")

    model, run_dir, best_pt, elapsed = train_model(data_yaml, epochs, run_name, batch=args.batch)
    print(f"[solve_q2] 训练完成,用时 {elapsed/60:.2f} 分钟,runs 目录 -> {run_dir}")
    print(f"[solve_q2] best.pt -> {best_pt}")

    from ultralytics import YOLO

    eval_model = YOLO(str(best_pt)) if best_pt.exists() else model

    print(f"[solve_q2] 在官方验证集上评估(N={len(test_stems)}),仅本次使用,不参与训练期选模型 ...")
    eval_out = evaluate_official(eval_model, test_stems)

    loss_curve = load_loss_curve(run_dir)

    result = {
        "seed": SEED,
        "img_size": IMG_SIZE,
        "epochs": epochs,
        "mode": "smoke" if args.smoke else "full",
        "class_names": CLASS_NAMES,
        "train_class_distribution": cls_count_train,
        "n_train_sub": len(train_stems),
        "n_val_sub": len(val_stems),
        "n_official_test": len(test_stems),
        "best_weights": str(best_pt),
        "run_dir": str(run_dir),
        "loss_curve": loss_curve,
        "per_class_metrics": eval_out["per_class_metrics"],
        "pr_curves": eval_out["pr_curves"],
        "dice_iou": eval_out["dice_iou"],
        "confusion": eval_out["confusion"],
        "metrics_table": eval_out["metrics_table"],
        "viz_samples": eval_out["viz_samples"],
        "train_seconds": round(elapsed, 1),
        "note": "分割真值为 bbox 反推的矩形伪掩码(数据集无逐像素多边形标注);Dice/mIoU 按此口径计算。",
    }
    joblib.dump(result, RESULT_PKL)
    print(f"[solve_q2] 结果已保存 -> {RESULT_PKL}")

    # ---- 打印汇总 ----
    print("\n===== 问题二 求解结果汇总(官方 413 验证集,全框口径)=====")
    header = f"{'class':>8s} {'AP@0.5':>8s} {'AP@.5:.95':>10s} {'AP_s':>7s} {'AP_m':>7s} {'AP_l':>7s} {'Dice':>7s} {'mIoU':>7s}"
    print(header)
    for row in result["metrics_table"]:
        print(
            f"{row['class']:>8s} {row['AP50']:8.4f} {row['AP50_95']:10.4f} "
            f"{row['AP_s']:7.4f} {row['AP_m']:7.4f} {row['AP_l']:7.4f} "
            f"{row['Dice']:7.4f} {row['mIoU']:7.4f}"
        )
    map50 = np.mean([r["AP50"] for r in result["metrics_table"]])
    map5095 = np.mean([r["AP50_95"] for r in result["metrics_table"]])
    print(f"{'mAP':>8s} {map50:8.4f} {map5095:10.4f}")

    print("\n混淆矩阵(行=真实, 列=预测; 末行/末列=背景/漏检/误检):")
    names = [CLASS_NAMES[c] for c in range(NUM_CLASSES)] + ["BG"]
    print(" " * 8 + "".join(f"{n:>8s}" for n in names))
    for i, row in enumerate(result["confusion"]):
        print(f"{names[i]:>8s}" + "".join(f"{v:8d}" for v in row))

    print(f"\n[solve_q2] ultralytics 训练产物(曲线图/权重/日志等)另存于 -> {run_dir}")
    print("[solve_q2] 完成。")


if __name__ == "__main__":
    main()
