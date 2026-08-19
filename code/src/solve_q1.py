# -*- coding: utf-8 -*-
"""问题一:残损分类的检测归约 —— 合成负样本验证 + 阈值 τ 扫描。

===== 方法 =====
问题一(判断一张图是否存在残损)被约化成基于问题二检测器的判定:
对图像跑 solve_q2.py 训练好的检测器(权重不变,不重新训练/微调),
取 S(I) = max_i s_i(该图全部预测框里的最高置信度,无检测记 0),
f(I) = 1{S(I) >= τ} 判定"有残损"。

这个约化是否成立,取决于"确定无残损的图像上,S(I) 确实普遍低于 τ"——
数据集里没有现成的纯负样本,所以用 syn_neg_rules.py / syn_neg_screen.py 从
残损图里挖不含标注框的干净区域拼接出的合成负样本代替。

===== 数据划分(严格不重叠,防泄漏)=====
- 调参集(tune):训练期验证正样本 330 张(与 solve_q2.py 的 10% 划分完全一致,
  直接复用 build_seg_dataset 保证可复现)+ 合成负样本 tune 半区(筛掉 tier1_high
  高危后剩余部分)。只在这个集合上扫描 τ,选 τ*。
- 评估集(eval):官方验证集正样本 413 张(与 solve_q2.py 最终评估用的完全一致)
  + 合成负样本 eval 半区(同样筛掉 tier1_high)。只用来在 τ* 下报告最终指标,
  不参与选 τ,避免选择泄漏。

===== 合成负样本的自动筛选口径 =====
syn_neg_screen.py 已经用跟检测模型完全独立的信号(源图黑名单 + HSV 天空启发式)
把 1000 张合成负样本分成 tier1_high(高危,225 张)/ tier2_mid(中危,386 张)/
tier3_low(低危,389 张)三级(见 code/src/noted.md 第 3 节)。本脚本自动排除
tier1_high(命中人工黑名单源图,或天空占比 >=0.5、大概率整块背景/天空而非残损
材质表面),保留 tier2_mid + tier3_low 作为可用负样本——这一步是"自动规则筛一遍,
先过滤掉明显有问题的",tier2_mid 尚未经过逐张人工复核,残留噪声属已知局限,
在 noted.md 第 4 节已有详细讨论,论文里应如实报告。
"""
import argparse
import csv
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve

from solve_q2 import (
    BATCH,
    CONF_EVAL,
    DATA_DIR,
    EVAL_CHUNK,
    IMG_SIZE,
    IOU_NMS_EVAL,
    RESULT_PKL as Q2_RESULT_PKL,
    SEED,
    SEG_DATASET,
    WORKERS,
    build_seg_dataset,
)

RESULT_PKL = DATA_DIR / "q1_result.pkl"
MANIFEST_CSV = DATA_DIR / "syn_neg_screen" / "manifest.csv"
EXCLUDE_TIERS = {"tier1_high"}  # 黑名单命中 / 天空占比>=0.5,排除出可用负样本池
TAU_GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)  # 0.05, 0.10, ..., 0.95


# ==========================================================================
# 1. 合成负样本:读 manifest,按 tier 过滤,按原始 tune/eval 划分分组
# ==========================================================================
def load_neg_manifest(smoke: bool = False):
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"找不到 {MANIFEST_CSV},请先跑 syn_neg_rules.py + syn_neg_screen.py")
    tune_neg, eval_neg = [], []
    excluded = 0
    with MANIFEST_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["orig_index"])
            if row["tier"] in EXCLUDE_TIERS:
                excluded += 1
                continue
            (tune_neg if idx < 500 else eval_neg).append(row["orig_path"])
    if smoke:
        tune_neg, eval_neg = tune_neg[:10], eval_neg[:10]
    return tune_neg, eval_neg, excluded


# ==========================================================================
# 2. 打分:S(I) = 该图全部预测框最高置信度(无检测记 0),分块推理避免大批量占显存
# ==========================================================================
def score_images(model, img_paths: list[str]) -> np.ndarray:
    scores = np.zeros(len(img_paths), dtype=np.float32)
    for start in range(0, len(img_paths), EVAL_CHUNK):
        chunk = img_paths[start:start + EVAL_CHUNK]
        results = model.predict(
            source=chunk,
            imgsz=IMG_SIZE,
            batch=BATCH,
            workers=WORKERS,
            conf=CONF_EVAL,
            iou=IOU_NMS_EVAL,
            verbose=False,
            retina_masks=False,  # 分类归约只用框置信度,不需要掩码,顺带避开 retina_masks 显存累积问题
            save=False,
        )
        for i, res in enumerate(results):
            if res.boxes is not None and len(res.boxes) > 0:
                scores[start + i] = float(res.boxes.conf.max().item())
        del results
        torch.cuda.empty_cache()
    return scores


# ==========================================================================
# 3. 主流程
# ==========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="冒烟测试:每组只取 10 张,验证全流程可跑通")
    args = parser.parse_args()

    if not Q2_RESULT_PKL.exists():
        print(f"[solve_q1] 找不到 {Q2_RESULT_PKL},请先跑 solve_q2.py 训练检测器")
        sys.exit(1)
    q2_result = joblib.load(Q2_RESULT_PKL)
    best_pt = q2_result["best_weights"]
    print(f"[solve_q1] mode={'SMOKE' if args.smoke else 'FULL'}  复用 q2 检测器权重(不微调) -> {best_pt}")

    _, _, val_stems, test_stems = build_seg_dataset(smoke=args.smoke)
    tune_neg, eval_neg, n_excluded = load_neg_manifest(smoke=args.smoke)
    print(f"[solve_q1] 训练期验证正样本(tune 正)={len(val_stems)}  官方验证集正样本(eval 正)={len(test_stems)}")
    print(f"[solve_q1] 合成负样本:排除 tier1_high 共 {n_excluded} 张后,tune 负={len(tune_neg)}  eval 负={len(eval_neg)}")

    val_paths = [str((SEG_DATASET / "images" / "train" / f"{s}.jpg").resolve()) for s in val_stems]
    test_paths = [str((SEG_DATASET / "images" / "test" / f"{s}.jpg").resolve()) for s in test_stems]

    tune_paths = val_paths + tune_neg
    tune_labels = np.array([1] * len(val_paths) + [0] * len(tune_neg))
    eval_paths = test_paths + eval_neg
    eval_labels = np.array([1] * len(test_paths) + [0] * len(eval_neg))

    from ultralytics import YOLO

    model = YOLO(str(best_pt))

    print(f"[solve_q1] 对调参集推理(N={len(tune_paths)}) ...")
    tune_scores = score_images(model, tune_paths)
    print(f"[solve_q1] 对评估集推理(N={len(eval_paths)}) ...")
    eval_scores = score_images(model, eval_paths)

    # ---- 调参集:扫描 τ,按 F1 选 τ* ----
    f1_tune = []
    for tau in TAU_GRID:
        pred = (tune_scores >= tau).astype(int)
        f1_tune.append(f1_score(tune_labels, pred, zero_division=0))
    f1_tune = np.array(f1_tune)
    tau_star = float(TAU_GRID[int(np.argmax(f1_tune))])
    print(f"[solve_q1] τ* = {tau_star:.2f} (调参集 F1={f1_tune.max():.4f})")

    # ---- 评估集:τ* 下报告 Acc/P/R/F1,S(I) 连续分数报告 AUC(与 τ 无关) ----
    eval_pred = (eval_scores >= tau_star).astype(int)
    acc = accuracy_score(eval_labels, eval_pred)
    prec = precision_score(eval_labels, eval_pred, zero_division=0)
    rec = recall_score(eval_labels, eval_pred, zero_division=0)
    f1 = f1_score(eval_labels, eval_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(eval_labels, eval_scores)
    auc_val = auc(fpr, tpr)

    result = {
        "seed": SEED,
        "img_size": IMG_SIZE,
        "mode": "smoke" if args.smoke else "full",
        "best_weights": str(best_pt),
        "tau_grid": TAU_GRID.tolist(),
        "f1_tune": [round(float(v), 4) for v in f1_tune],
        "tau_star": round(tau_star, 2),
        "tune_n_pos": len(val_paths),
        "tune_n_neg": len(tune_neg),
        "eval_n_pos": len(test_paths),
        "eval_n_neg": len(eval_neg),
        "eval_metrics": {
            "acc": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "auc": round(float(auc_val), 4),
        },
        "roc": {"fpr": np.round(fpr, 4).tolist(), "tpr": np.round(tpr, 4).tolist()},
        "neg_sample_note": (
            f"合成负样本自动排除 tier1_high(黑名单命中/天空占比>=0.5)共 {n_excluded} 张;"
            "tier2_mid 未经逐张人工复核,残留噪声属已知局限(见 code/src/noted.md 第 4 节)。"
        ),
        "note": "检测器权重复用 solve_q2.py 训练结果,本脚本不重新训练/微调,只做推理+阈值扫描。",
    }
    joblib.dump(result, RESULT_PKL)
    print(f"[solve_q1] 结果已保存 -> {RESULT_PKL}")

    print("\n===== 问题一 求解结果汇总 =====")
    print(f"τ* = {tau_star:.2f}(调参集 N={len(tune_paths)}, 正{len(val_paths)}/负{len(tune_neg)}, F1={f1_tune.max():.4f})")
    print(f"评估集(N={len(eval_paths)}, 正{len(test_paths)}/负{len(eval_neg)})在 τ* 下:")
    print(f"  Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  AUC={auc_val:.4f}")
    print("[solve_q1] 完成。")


if __name__ == "__main__":
    main()
