# -*- coding: utf-8 -*-
"""诊断脚本(不进正式三家对比流程,只是抽查用)。

===== 动机 =====
三家方法在合成负样本(2x2 拼接)上的评估结果——检测归约 AUC=0.8435 < SVM
AUC=0.9951 < CNN AUC=0.9999——分数随模型容量单调上升、逼近满分。肉眼抽查合成
负样本发现普遍带硬拼接缝/旋转裁剪留下的黑色三角形留白,怀疑 SVM/CNN 的高分
不是在测"残损识别能力",而是在测"能不能认出这是拼接图"这个更简单、跟题目
本意无关的任务。

本脚本用从 Wikimedia Commons 下载并人工筛过的真实集装箱照片(单张连续照片,
不拼接,肉眼确认无明显残损)替换合成负样本,正样本仍用官方验证集 413 张,
复用三家已经训练好的模型/权重(不重新训练),重新评估一遍,看分数是否回落。
真实负样本的框架/取景与训练分布不完全一致(多为集装箱堆场中/远景,不是训练集
里那种贴近表面的特写),这是初筛条件下的局限,结果只作方向性诊断,不作为
正式指标写入 tab:q3-compare。
"""
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
import numpy as np
import torch
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve

from solve_q1_svm import extract_features
from solve_q1_cnn import EVAL_TF, build_model
from solve_q2 import DATA_DIR, SEG_DATASET, build_seg_dataset

REAL_NEG_DIR = DATA_DIR / "real_neg"


def eval_metrics_from(labels, scores, tau):
    pred = (scores >= tau).astype(int)
    acc = accuracy_score(labels, pred)
    prec = precision_score(labels, pred, zero_division=0)
    rec = recall_score(labels, pred, zero_division=0)
    f1 = f1_score(labels, pred, zero_division=0)
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_val = auc(fpr, tpr)
    return {
        "acc": round(float(acc), 4), "precision": round(float(prec), 4),
        "recall": round(float(rec), 4), "f1": round(float(f1), 4), "auc": round(float(auc_val), 4),
    }


def main():
    real_neg_paths = sorted(str(p) for p in REAL_NEG_DIR.glob("*.jpg"))
    print(f"[realneg_check] 真实负样本 N={len(real_neg_paths)}")
    if len(real_neg_paths) < 10:
        print(f"[realneg_check] {REAL_NEG_DIR} 里真实负样本数量太少,先跑筛选脚本把图放进去")
        sys.exit(1)

    _, _, val_stems, test_stems = build_seg_dataset(smoke=False)
    eval_pos_paths = [str((SEG_DATASET / "images" / "test" / f"{s}.jpg").resolve()) for s in test_stems]
    labels = np.array([1] * len(eval_pos_paths) + [0] * len(real_neg_paths))
    all_paths = eval_pos_paths + real_neg_paths
    print(f"[realneg_check] 评估集: 正={len(eval_pos_paths)}(官方验证集,不变)  负={len(real_neg_paths)}(真实照片)")

    results = {}

    # ---- 1. 检测归约:复用 q2 检测器权重,τ* 用 q1_result.pkl 里已选好的 ----
    from solve_q2 import RESULT_PKL as Q2_RESULT_PKL
    from solve_q1 import score_images
    from ultralytics import YOLO

    q1_result = joblib.load(DATA_DIR / "q1_result.pkl")
    tau_star_det = q1_result["tau_star"]
    q2_result = joblib.load(Q2_RESULT_PKL)
    model = YOLO(str(q2_result["best_weights"]))
    print(f"[realneg_check] 检测归约推理(τ*={tau_star_det:.2f}) ...")
    det_scores = score_images(model, all_paths)
    results["detection_reduction"] = eval_metrics_from(labels, det_scores, tau_star_det)
    del model
    torch.cuda.empty_cache()

    # ---- 2. SVM:复用已训练好的 svc+scaler,τ* 用 q1_svm_result.pkl 里已选好的 ----
    svm_bundle = joblib.load(DATA_DIR / "q1_svm_model.joblib")
    svc, scaler = svm_bundle["svc"], svm_bundle["scaler"]
    svm_result = joblib.load(DATA_DIR / "q1_svm_result.pkl")
    tau_star_svm = svm_result["tau_star"]
    print(f"[realneg_check] SVM 特征提取 + 推理(τ*={tau_star_svm:.2f}) ...")
    feats = np.stack([extract_features(cv2.imread(p)) for p in all_paths])
    feats_s = scaler.transform(feats)
    svm_proba = svc.predict_proba(feats_s)[:, 1]
    results["svm"] = eval_metrics_from(labels, svm_proba, tau_star_svm)

    # ---- 3. CNN:复用已训练好的权重,τ* 用 q1_cnn_result.pkl 里已选好的 ----
    cnn_result = joblib.load(DATA_DIR / "q1_cnn_result.pkl")
    tau_star_cnn = cnn_result["tau_star"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = build_model()
    cnn_model.load_state_dict(torch.load(DATA_DIR / "q1_cnn_best.pt", map_location=device))
    cnn_model.eval()
    print(f"[realneg_check] CNN 推理(τ*={tau_star_cnn:.2f}) ...")
    cnn_logits = []
    with torch.no_grad():
        for i in range(0, len(all_paths), 32):
            batch_paths = all_paths[i:i + 32]
            imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in batch_paths]
            x = torch.stack([EVAL_TF(im) for im in imgs]).to(device)
            logits = cnn_model(x).squeeze(1)
            cnn_logits.append(logits.cpu().numpy())
    cnn_logits = np.concatenate(cnn_logits)
    cnn_proba = 1 / (1 + np.exp(-cnn_logits))
    results["cnn"] = eval_metrics_from(labels, cnn_proba, tau_star_cnn)

    out = DATA_DIR / "q1_realneg_check_result.pkl"
    joblib.dump({
        "n_pos": len(eval_pos_paths), "n_neg": len(real_neg_paths), "results": results,
        "note": "诊断用,真实负样本未做取景/尺度匹配筛选,不进正式对比表,只判断合成负样本拼接伪影是否影响三家分数。",
    }, out)
    print(f"[realneg_check] 结果已保存 -> {out}")
    print(f"\n===== 真实负样本(N={len(real_neg_paths)},非拼接)重新评估 =====")
    for name, m in results.items():
        print(f"  {name:20s} Acc={m['acc']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}"
              f"  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")


if __name__ == "__main__":
    main()
