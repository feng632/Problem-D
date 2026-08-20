# -*- coding: utf-8 -*-
"""Q3 消融实验(逐项累加,FILL_CHECKLIST 正式条目):
自基准配置起依次加入 类别加权 / P2头 / 1280输入 / 强增强,每步只变一个变量。

Step4(全部加完 = 类别加权+P2头+1280输入+强增强)跟 explore_q3_s_res.py 跑过的
explore_s_b2_i1280_e30 是同一套配置(P2+cls_pw=1.0+强增强,只是 imgsz 从模块级
变量猴子补丁成了 1280),结果可以直接复用,不用再跑一次,所以本脚本只覆盖
Step0~3:

  step0  基准         : 无P2  cls_pw=0  imgsz=640   弱增强
  step1  +类别加权     : 无P2  cls_pw=1  imgsz=640   弱增强
  step2  +P2头        : 有P2  cls_pw=1  imgsz=640   弱增强
  step3  +1280输入     : 有P2  cls_pw=1  imgsz=1280  弱增强
  (step4 +强增强,复用 explore_s_b2_i1280_e30,不在本脚本里跑)

跑法(尚未执行,先准备代码):
  python ablate_q3_steps.py --step 0 --batch 2 --epochs 30
  python ablate_q3_steps.py --step 1 --batch 2 --epochs 30
  python ablate_q3_steps.py --step 2 --batch 2 --epochs 30
  python ablate_q3_steps.py --step 3 --batch 2 --epochs 30

结果各自存 code/data/q3_ablate_step{N}.pkl,mAP@0.5 直接从 ultralytics
results.csv 最后一行取(metrics/mAP50(B));AP_s(小目标 AP)需要额外跑一遍
官方 413 张测试集评估(evaluate_official 里已有按面积分桶的 AP_s/m/l),
本脚本训练完之后顺带跑一次,一并存进同一个 pkl。
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

# step -> (use_p2, cls_pw, imgsz, strong_aug);strong_aug 固定 False,
# step4(强增强=True)不在这张表里,直接复用 explore_s_b2_i1280_e30。
STEP_CONFIGS = {
    0: dict(use_p2=False, cls_pw=0.0, imgsz=640, strong_aug=False, label="基准"),
    1: dict(use_p2=False, cls_pw=1.0, imgsz=640, strong_aug=False, label="+类别加权"),
    2: dict(use_p2=True, cls_pw=1.0, imgsz=640, strong_aug=False, label="+P2头"),
    3: dict(use_p2=True, cls_pw=1.0, imgsz=1280, strong_aug=False, label="+1280输入"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int, required=True, choices=sorted(STEP_CONFIGS))
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--name", type=str, default=None, help="默认 ablate_step{N}")
args = parser.parse_args()

cfg = STEP_CONFIGS[args.step]
run_name = args.name or f"ablate_step{args.step}"
solve_q2.IMG_SIZE = cfg["imgsz"]  # train_model()/evaluate_official() 都读这个模块级变量


def main():
    import joblib

    data_yaml, train_stems, val_stems, test_stems = solve_q2.build_seg_dataset(smoke=False)
    print(
        f"[ablate] step={args.step}({cfg['label']}) use_p2={cfg['use_p2']} "
        f"cls_pw={cfg['cls_pw']} imgsz={cfg['imgsz']} strong_aug={cfg['strong_aug']} "
        f"batch={args.batch} epochs={args.epochs}"
    )

    t0 = time.time()
    model, run_dir, best_pt, elapsed = solve_q2.train_model(
        data_yaml,
        args.epochs,
        run_name,
        batch=args.batch,
        use_p2=cfg["use_p2"],
        cls_pw=cfg["cls_pw"],
        strong_aug=cfg["strong_aug"],
    )
    print(f"[ablate] 训练完成 用时 {elapsed/60:.2f} 分钟  run_dir={run_dir}  best={best_pt}")

    # 官方 413 张测试集评估,拿 mAP@0.5 / AP_s(小目标)口径统一的数字
    # (跟 results.csv 里训练期验证集的 mAP50(B) 不是同一份数据,消融表要用
    # 哪一个由建模手定,这里两个都存)。
    eval_result = solve_q2.evaluate_official(model, test_stems)

    import gc
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()

    out = solve_q2.DATA_DIR / f"q3_ablate_step{args.step}.pkl"
    joblib.dump(
        {
            "step": args.step,
            "label": cfg["label"],
            "use_p2": cfg["use_p2"],
            "cls_pw": cfg["cls_pw"],
            "imgsz": cfg["imgsz"],
            "strong_aug": cfg["strong_aug"],
            "batch": args.batch,
            "epochs": args.epochs,
            "train_seconds": round(elapsed, 1),
            "run_dir": str(run_dir),
            "best_pt": str(best_pt),
            "eval_official": eval_result,
            "note": "Q3 消融实验正式条目(逐项累加),step4(+强增强)复用 explore_s_b2_i1280_e30。",
        },
        out,
    )
    print(f"[ablate] 结果已保存 -> {out}")


if __name__ == "__main__":
    main()
