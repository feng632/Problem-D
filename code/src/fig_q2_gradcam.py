# -*- coding: utf-8 -*-
"""Q3 Grad-CAM 可视化(FILL_CHECKLIST 正式条目:对验证集样例生成激活热力图,
3 类各 1-2 张)。

不用自己写 hook/backward——ultralytics 8.4.120 自带的
`class_activation_map()`(见 utils/plotting.py)就是 LayerCAM 实现:
hook Segment 头的输入(多尺度 feats)和输出(每尺度拼接后的原始 class
logits),对每张图选中的类别做反传,ReLU 后跨尺度取 max 融合、归一化、
叠加到原图,直接存 `<stem>_cam.jpg`。用 `model.predict(..., visualize=True,
classes=[目标类])` 就能触发,不用改 ultralytics 源码。

权重用 code/data/runs_q2/full/weights/best.pt(官方 100-epoch 训练出的最终
模型,不是 explore 系列的 30-epoch 探索性权重)。
样例图从"训练期验证集"(330 张,build_seg_dataset 里划出来的那部分,
build_seg_dataset() 用固定 SEED=42,划分可复现)里按类别筛,每类挑最多 2 张
标注里包含该类别的图。
"""
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))
import solve_q2

N_PER_CLASS = 2
CONF = 0.25
GRADCAM_DIR = solve_q2.DATA_DIR / "runs_q2" / "gradcam"


def stems_with_class(stems, target_cls):
    out = []
    for s in stems:
        lbl = solve_q2.SEG_DATASET / "labels" / "train" / f"{s}.txt"
        if not lbl.exists():
            continue
        with open(lbl) as f:
            classes_in_img = {int(line.split()[0]) for line in f if line.strip()}
        if target_cls in classes_in_img:
            out.append(s)
    return out


def main():
    best_weights = solve_q2.RUNS_DIR / "full" / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"官方 full 训练权重不存在: {best_weights}")

    _, _, val_stems, _ = solve_q2.build_seg_dataset(smoke=False)

    from ultralytics import YOLO

    model = YOLO(str(best_weights))

    saved = []
    for cls_id, cls_name in solve_q2.CLASS_NAMES.items():
        cands = stems_with_class(val_stems, cls_id)
        picked = cands[:N_PER_CLASS]
        print(f"[gradcam] class={cls_id}({cls_name})  验证集含该类的图共 {len(cands)} 张,取 {len(picked)} 张: {picked}")
        for stem in picked:
            img_path = solve_q2.SEG_DATASET / "images" / "train" / f"{stem}.jpg"
            run_name = f"{stem}_cls{cls_id}_{cls_name}"
            model.predict(
                source=str(img_path),
                visualize=True,
                classes=[cls_id],
                conf=CONF,
                imgsz=solve_q2.IMG_SIZE,
                project=str(GRADCAM_DIR),
                name=run_name,
                exist_ok=True,
                save=False,
                verbose=False,
            )
            out_jpg = GRADCAM_DIR / run_name / f"{stem}_cam.jpg"
            saved.append((cls_id, cls_name, stem, str(out_jpg), out_jpg.exists()))

    print("\n[gradcam] 汇总:")
    for cls_id, cls_name, stem, path, ok in saved:
        print(f"  class={cls_id}({cls_name}) stem={stem} -> {path}  存在={ok}")

    n_ok = sum(1 for *_, ok in saved if ok)
    print(f"\n[gradcam] 完成 {n_ok}/{len(saved)} 张热力图,输出目录 -> {GRADCAM_DIR}")


if __name__ == "__main__":
    main()
