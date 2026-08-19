# -*- coding: utf-8 -*-
"""问题一(三家对比之一):自建小 CNN 分类器(ResNet18 部分微调)。

===== 方法与诚实说明 =====
不复用问题二的检测器,单独训练一个二分类网络直接回答"图像是否存在残损"。
骨干用 ImageNet 预训练的 ResNet18,冻结 layer1-3,只微调 layer4 + 新接的
线性分类头。训练池只有 ~3300 张图,从零随机初始化一个 CNN 架构在这个规模下
大概率过拟合/学不出泛化特征;部分微调预训练特征是这个数据规模下更稳的工程
选择。这里如实说明该设计取舍,不冒充"完全从零设计架构训练"。

===== 数据划分(与 solve_q1_svm.py 用同一个训练池,防泄漏原则一致)=====
训练池:solve_q2.py 90% 训练划分的正样本(train_sub_stems,~2970 张)+ 调参半
合成负样本(与 solve_q1.py 完全一致的 tune_neg,~375 张)。
训练池内部再按 85/15 分层切出 cnn_train / cnn_val——cnn_val 只用来做 epoch
早停和选阈值 τ*,不碰评估集。正负比例失衡(~8:1,正样本占多数,与检测任务里
"正样本稀少"的常见失衡方向相反),训练时用 WeightedRandomSampler 让每个
batch 里两类样本数大致相等。
评估池与 solve_q1.py / solve_q1_svm.py 完全一致:官方验证集正样本(413 张)+
评估半合成负样本,只在最终选定的权重与 τ* 下评估一次。
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
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from solve_q1 import load_neg_manifest
from solve_q2 import DATA_DIR, SEED, SEG_DATASET, build_seg_dataset

RESULT_PKL = DATA_DIR / "q1_cnn_result.pkl"
TAU_GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)
IMG_SIZE_CNN = 224
EPOCHS = 15
SMOKE_EPOCHS = 2
BATCH = 32
PATIENCE = 5
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMG_SIZE_CNN),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
EVAL_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE_CNN, IMG_SIZE_CNN)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ==========================================================================
# 1. Dataset / 模型
# ==========================================================================
class DamageDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths = list(paths)
        self.labels = labels
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.tf(img)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for name, p in m.named_parameters():
        p.requires_grad = name.startswith("layer4") or name.startswith("fc")
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m.to(DEVICE)


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss, all_logits, all_labels = 0.0, [], []
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss / len(loader.dataset), logits, labels


# ==========================================================================
# 2. 主流程
# ==========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="冒烟测试:训练/评估池都用小子集,epochs 也减少")
    args = parser.parse_args()
    epochs = SMOKE_EPOCHS if args.smoke else EPOCHS
    print(f"[solve_q1_cnn] mode={'SMOKE' if args.smoke else 'FULL'}  epochs={epochs}  device={DEVICE}")

    _, train_stems, val_stems, test_stems = build_seg_dataset(smoke=args.smoke)
    tune_neg, eval_neg, n_excluded = load_neg_manifest(smoke=args.smoke)
    print(f"[solve_q1_cnn] 训练正样本(q2 90% 训练划分)={len(train_stems)}  调参半负样本={len(tune_neg)}")
    print(f"[solve_q1_cnn] 评估集: 正={len(test_stems)}  负={len(eval_neg)}(已排除 tier1_high 共 {n_excluded} 张)")

    train_pos_paths = [str((SEG_DATASET / "images" / "train" / f"{s}.jpg").resolve()) for s in train_stems]
    eval_pos_paths = [str((SEG_DATASET / "images" / "test" / f"{s}.jpg").resolve()) for s in test_stems]

    pool_paths = np.array(train_pos_paths + tune_neg)
    pool_labels = np.array([1] * len(train_pos_paths) + [0] * len(tune_neg))
    eval_paths = eval_pos_paths + eval_neg
    eval_labels = np.array([1] * len(eval_pos_paths) + [0] * len(eval_neg))

    cnn_train_p, cnn_val_p, cnn_train_y, cnn_val_y = train_test_split(
        pool_paths, pool_labels, test_size=0.15, stratify=pool_labels, random_state=SEED
    )
    print(f"[solve_q1_cnn] 训练池内部划分: cnn_train={len(cnn_train_p)}  cnn_val={len(cnn_val_p)}")

    train_ds = DamageDataset(cnn_train_p, cnn_train_y, TRAIN_TF)
    val_ds = DamageDataset(cnn_val_p, cnn_val_y, EVAL_TF)
    eval_ds = DamageDataset(eval_paths, eval_labels, EVAL_TF)

    class_count = np.bincount(cnn_train_y)
    sample_weight = 1.0 / class_count[cnn_train_y]
    sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model()
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if n.startswith("layer4") and p.requires_grad], "lr": 1e-5},
        {"params": [p for n, p in model.named_parameters() if n.startswith("fc")], "lr": 1e-3},
    ]
    optimizer = torch.optim.Adam(param_groups)

    best_f1, best_state, patience_ctr = -1.0, None, 0
    loss_curve = []
    for epoch in range(epochs):
        train_loss, _, _ = run_epoch(model, train_loader, optimizer)
        val_loss, val_logits, val_y = run_epoch(model, val_loader)
        val_proba = 1 / (1 + np.exp(-val_logits))
        val_f1 = f1_score(val_y, (val_proba >= 0.5).astype(int), zero_division=0)
        loss_curve.append({
            "epoch": epoch + 1, "train_loss": round(float(train_loss), 4),
            "val_loss": round(float(val_loss), 4), "val_f1@0.5": round(float(val_f1), 4),
        })
        print(f"[solve_q1_cnn] epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}"
              f"  val_loss={val_loss:.4f}  val_f1@0.5={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"[solve_q1_cnn] 早停(patience={PATIENCE})")
                break

    model.load_state_dict(best_state)

    # ---- cnn_val 上选 τ*(F1 最优),评估集只用一次 ----
    _, val_logits, val_y = run_epoch(model, val_loader)
    val_proba = 1 / (1 + np.exp(-val_logits))
    f1_tune = np.array([f1_score(val_y, (val_proba >= t).astype(int), zero_division=0) for t in TAU_GRID])
    tau_star = float(TAU_GRID[int(np.argmax(f1_tune))])
    print(f"[solve_q1_cnn] τ* = {tau_star:.2f}(cnn_val F1={f1_tune.max():.4f})")

    _, eval_logits, eval_y_out = run_epoch(model, eval_loader)
    eval_proba = 1 / (1 + np.exp(-eval_logits))
    eval_pred = (eval_proba >= tau_star).astype(int)
    acc = accuracy_score(eval_y_out, eval_pred)
    prec = precision_score(eval_y_out, eval_pred, zero_division=0)
    rec = recall_score(eval_y_out, eval_pred, zero_division=0)
    f1 = f1_score(eval_y_out, eval_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(eval_y_out, eval_proba)
    auc_val = auc(fpr, tpr)

    weights_path = DATA_DIR / "q1_cnn_best.pt"
    torch.save(best_state, weights_path)

    result = {
        "seed": SEED,
        "mode": "smoke" if args.smoke else "full",
        "backbone": "resnet18(ImageNet预训练,冻结layer1-3,微调layer4+fc)",
        "epochs_run": len(loss_curve),
        "loss_curve": loss_curve,
        "tau_grid": TAU_GRID.tolist(),
        "f1_tune": [round(float(v), 4) for v in f1_tune],
        "tau_star": round(tau_star, 2),
        "pool_n_pos": len(train_pos_paths),
        "pool_n_neg": len(tune_neg),
        "cnn_train_n": len(cnn_train_p),
        "cnn_val_n": len(cnn_val_p),
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
        "weights_path": str(weights_path),
        "note": (
            "训练/内部验证/评估三段式与 solve_q1.py 保持同一防泄漏原则:τ* 只在训练池内部切出的"
            "cnn_val 上选,评估集(413 正 + 评估半负样本)只在最终 τ* 下用一次,不参与任何选择过程。"
            "骨干为部分微调的预训练 ResNet18(冻结 layer1-3),不是从零设计训练的架构,如实说明。"
        ),
    }
    joblib.dump(result, RESULT_PKL)
    print(f"[solve_q1_cnn] 结果已保存 -> {RESULT_PKL}")
    print(f"[solve_q1_cnn] 评估集(N={len(eval_paths)})在 τ*={tau_star:.2f} 下: "
          f"Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  AUC={auc_val:.4f}")
    print("[solve_q1_cnn] 完成。")


if __name__ == "__main__":
    main()
