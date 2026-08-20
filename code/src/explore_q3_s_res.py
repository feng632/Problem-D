# -*- coding: utf-8 -*-
"""Q3 探索性实验(不是 FILL_CHECKLIST 正式条目,是效率调优前测,结果不进 tab:q3-eff)。

背景:resolution_sensitivity 测出一个在 imgsz=1280 训练的模型,推理时缩到 960
反而 mAP50 更高(0.3993 > 0.3623)。这个脚本测的是另一件事——如果训练本身就用
960(而不是训练 1280、推理再缩),配合把显存压到 6GB 以内用的 batch=2,精度和
耗时分别是什么样,给"要不要把训练分辨率也换成 960"这个决定提供数据。

直接复用 solve_q2.py 的 build_seg_dataset() / train_model()(同一套数据划分、
同一套增强超参、同一个 P2 结构),只在调用前把模块级 IMG_SIZE 猴子补丁成想测的
分辨率,不复制一份超参逻辑出来,避免跟正式流程的配置产生分叉/漂移。
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

sys.path.insert(0, str(Path(__file__).parent))
import solve_q2

parser = argparse.ArgumentParser()
parser.add_argument("--imgsz", type=int, required=True)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--name", type=str, required=True)
args = parser.parse_args()

solve_q2.IMG_SIZE = args.imgsz  # train_model() 内部读的是模块级这个全局变量

def main():
    import joblib

    data_yaml, train_stems, val_stems, test_stems = solve_q2.build_seg_dataset(smoke=False)
    print(f"[explore] name={args.name} imgsz={args.imgsz} batch={args.batch} epochs={args.epochs}")
    print(f"[explore] train_sub={len(train_stems)}  val_sub={len(val_stems)}  official_test(未用)={len(test_stems)}")

    t0 = time.time()
    model, run_dir, best_pt, elapsed = solve_q2.train_model(
        data_yaml, args.epochs, args.name, batch=args.batch
    )
    print(f"[explore] 完成 name={args.name}  用时 {elapsed/60:.2f} 分钟  run_dir={run_dir}  best={best_pt}")

    import gc
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()

    out = solve_q2.DATA_DIR / f"q3_explore_{args.name}.pkl"
    joblib.dump({
        "name": args.name, "imgsz": args.imgsz, "batch": args.batch, "epochs": args.epochs,
        "train_seconds": round(elapsed, 1), "run_dir": str(run_dir), "best_pt": str(best_pt),
        "note": "探索性实验(batch=2 显存预算下 960 vs 1280 训练分辨率对比),不进 tab:q3-eff 正式表。",
    }, out)
    print(f"[explore] 结果已保存 -> {out}")

if __name__ == "__main__":
    main()
