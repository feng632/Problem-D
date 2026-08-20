# -*- coding: utf-8 -*-
"""问题一(三家对比之一):传统特征 + SVM 基线。

===== 方法 =====
不复用问题二的检测器,从零训练一个"手工特征 + SVM"的二分类器,直接回答
"图像是否存在残损"这一图像级问题。特征拼接四类:
  - HOG(方向梯度直方图):抓边缘/形状分布,残损区域(凹陷边缘、破洞边界)
    通常带来局部梯度方向的突变
  - LBP(局部二值模式):抓局部纹理粗糙度,锈蚀/破损表面纹理与完好金属面不同
  - HSV 颜色直方图:抓整体色彩分布,锈蚀常见的橙红色调是一个弱但有用的信号
  - GLCM(灰度共生矩阵)统计量(对比度/同质性/能量/相关性):抓更整体的纹理
    规律性,残损区域通常比完好表面更不规则
这四类拼成一个定长向量,标准化后喂给 RBF-SVM。

===== 数据划分(与 solve_q1.py 同一防泄漏原则,但训练池不同)=====
solve_q1.py 的检测归约方案不需要训练(直接复用 q2 检测器权重),它的"调参集"
只是用来选阈值 τ,所以只用了 q2 训练期验证正样本(330 张)。
SVM 是真的要训练分类器,需要更大的正样本池,因此训练池改用:
  - 训练正样本:solve_q2.py 90% 训练划分(build_seg_dataset 的 train_sub_stems,
    ~2970 张,即 q2 检测器实际用于学权重的那部分正样本)
  - 调参半合成负样本(与 solve_q1.py 完全一致的 tune_neg,~375 张,已排除
    tier1_high 高危样本)
SVM 超参(C, gamma)用训练池内部 3 折分层交叉验证选;决策阈值 τ 用同一训练池的
out-of-fold(交叉验证预测,不是同集合内验证)概率选,避免"训练集内选阈值"的
乐观偏差。
评估池与 solve_q1.py / solve_q3.py 完全一致:官方验证集正样本(413 张)+ 评估半
合成负样本,只在选定的 τ* 下算一次最终指标,不参与任何选择过程。
"""
import argparse
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
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from solve_q1 import load_neg_manifest
from solve_q2 import DATA_DIR, SEED, SEG_DATASET, build_seg_dataset

RESULT_PKL = DATA_DIR / "q1_svm_result.pkl"
TAU_GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)
FEAT_IMG_SIZE = 128  # 特征提取前统一缩放到的边长,兼顾速度与信息量


# ==========================================================================
# 1. 特征提取:HOG + LBP + HSV颜色直方图 + GLCM 拼成定长向量
# ==========================================================================
def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, (FEAT_IMG_SIZE, FEAT_IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(
        gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
        block_norm="L2-Hys", feature_vector=True,
    )

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_hist = np.concatenate(
        [cv2.calcHist([hsv], [i], None, [8], [0, 256]).flatten() for i in range(3)]
    )
    color_hist = color_hist / (color_hist.sum() + 1e-9)

    gray_q = (gray // 8).astype(np.uint8)  # 256 -> 32 灰度级,减小 GLCM 计算量
    glcm = graycomatrix(
        gray_q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32, symmetric=True, normed=True,
    )
    glcm_feat = np.concatenate(
        [graycoprops(glcm, prop).flatten() for prop in ("contrast", "homogeneity", "energy", "correlation")]
    )

    return np.concatenate([hog_feat, lbp_hist, color_hist, glcm_feat]).astype(np.float32)


def extract_features_batch(paths: list[str]) -> np.ndarray:
    feats = []
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        feats.append(extract_features(img))
        if (i + 1) % 500 == 0 or (i + 1) == len(paths):
            print(f"    特征提取 {i + 1}/{len(paths)}")
    return np.stack(feats)


# ==========================================================================
# 2. 主流程
# ==========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="冒烟测试:训练/评估池都用小子集")
    args = parser.parse_args()
    print(f"[solve_q1_svm] mode={'SMOKE' if args.smoke else 'FULL'}")

    _, train_stems, val_stems, test_stems = build_seg_dataset(smoke=args.smoke)
    tune_neg, eval_neg, n_excluded = load_neg_manifest(smoke=args.smoke)
    print(f"[solve_q1_svm] 训练正样本(q2 90% 训练划分)={len(train_stems)}  调参半负样本={len(tune_neg)}")
    print(f"[solve_q1_svm] 评估集: 正={len(test_stems)}  负={len(eval_neg)}(已排除 tier1_high 共 {n_excluded} 张)")

    train_pos_paths = [str((SEG_DATASET / "images" / "train" / f"{s}.jpg").resolve()) for s in train_stems]
    eval_pos_paths = [str((SEG_DATASET / "images" / "test" / f"{s}.jpg").resolve()) for s in test_stems]

    train_paths = train_pos_paths + tune_neg
    train_labels = np.array([1] * len(train_pos_paths) + [0] * len(tune_neg))
    eval_paths = eval_pos_paths + eval_neg
    eval_labels = np.array([1] * len(eval_pos_paths) + [0] * len(eval_neg))

    print(f"[solve_q1_svm] 提取训练池特征(N={len(train_paths)}) ...")
    X_train = extract_features_batch(train_paths)
    print(f"[solve_q1_svm] 提取评估集特征(N={len(eval_paths)}) ...")
    X_eval = extract_features_batch(eval_paths)
    print(f"[solve_q1_svm] 特征维度 = {X_train.shape[1]}")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_eval_s = scaler.transform(X_eval)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    # ---- 第一步:GridSearchCV 选 C/gamma(不开 probability,更快)----
    param_grid = {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.001]}
    print("[solve_q1_svm] GridSearchCV(C, gamma) 3 折交叉验证 ...")
    gs = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
        param_grid, cv=cv, scoring="f1", n_jobs=-1,
    )
    gs.fit(X_train_s, train_labels)
    best_params = gs.best_params_
    print(f"[solve_q1_svm] 最优超参: {best_params}  CV F1={gs.best_score_:.4f}")

    # ---- 第二步:用最优超参、开 probability=True 只重新拟合一次,拿到概率输出 ----
    final_svc = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED, **best_params)

    print("[solve_q1_svm] out-of-fold 概率(选阈值用,避免训练集内验证的乐观偏差) ...")
    oof_proba = cross_val_predict(final_svc, X_train_s, train_labels, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    f1_tune = np.array([f1_score(train_labels, (oof_proba >= t).astype(int), zero_division=0) for t in TAU_GRID])
    tau_star = float(TAU_GRID[int(np.argmax(f1_tune))])
    print(f"[solve_q1_svm] τ* = {tau_star:.2f}(训练池 out-of-fold F1={f1_tune.max():.4f})")

    print("[solve_q1_svm] 在完整训练池上拟合最终模型 ...")
    final_svc.fit(X_train_s, train_labels)

    eval_proba = final_svc.predict_proba(X_eval_s)[:, 1]
    eval_pred = (eval_proba >= tau_star).astype(int)
    acc = accuracy_score(eval_labels, eval_pred)
    prec = precision_score(eval_labels, eval_pred, zero_division=0)
    rec = recall_score(eval_labels, eval_pred, zero_division=0)
    f1 = f1_score(eval_labels, eval_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(eval_labels, eval_proba)
    auc_val = auc(fpr, tpr)

    model_path = DATA_DIR / "q1_svm_model.joblib"
    joblib.dump({"svc": final_svc, "scaler": scaler}, model_path)

    result = {
        "seed": SEED,
        "mode": "smoke" if args.smoke else "full",
        "feature_dim": int(X_train.shape[1]),
        "svm_best_params": best_params,
        "cv_f1": round(float(gs.best_score_), 4),
        "tau_grid": TAU_GRID.tolist(),
        "f1_tune": [round(float(v), 4) for v in f1_tune],
        "tau_star": round(tau_star, 2),
        "train_n_pos": len(train_pos_paths),
        "train_n_neg": len(tune_neg),
        "eval_n_pos": len(eval_pos_paths),
        "eval_n_neg": len(eval_neg),
        "eval_metrics": {
            "acc": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "auc": round(float(auc_val), 4),
        },
        "roc": {"fpr": np.round(fpr, 4).tolist(), "tpr": np.round(tpr, 4).tolist()},
        "model_path": str(model_path),
        "note": (
            "训练池(训练正样本 ~2970 + 调参半负样本)与评估池(官方验证正样本 413 + 评估半负样本)"
            "完全不重叠;C/gamma 由训练池内 3 折交叉验证选,τ* 由训练池 out-of-fold 概率选,"
            "评估集只在最终 τ* 下用一次,不参与任何选择过程。"
        ),
    }
    joblib.dump(result, RESULT_PKL)
    print(f"[solve_q1_svm] 结果已保存 -> {RESULT_PKL}")
    print(f"[solve_q1_svm] 评估集(N={len(eval_paths)})在 τ*={tau_star:.2f} 下: "
          f"Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  AUC={auc_val:.4f}")
    print("[solve_q1_svm] 完成。")


if __name__ == "__main__":
    main()
