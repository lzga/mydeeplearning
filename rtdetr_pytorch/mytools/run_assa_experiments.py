# -*- coding: utf-8 -*-
r"""
串行训练脚本：baseline → FB-C5 → baseline+ASSA → FB-C5+ASSA

四组实验串行运行，随机种子统一为 seed=4，确保公平对比。

运行方式：
  python mytools/run_assa_experiments.py

实验矩阵：
  实验0: baseline         (RT-DETR-R18 原始)
  实验1: FB-C5            (FasterBlock at C5 only)
  实验2: baseline+ASSA     (ASSA Top-K=0.75 in AIFI)
  实验3: FB-C5+ASSA       (FasterBlock C5 + ASSA Top-K=0.75)
"""

from __future__ import annotations

import os
import sys
import runpy
import random
import traceback
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr


PROJECT_ROOT = Path(r"D:\Learn\RTDETR\RT-DETR-main\rtdetr_pytorch")
RTDETR_PYTHON = Path(r"D:\Learn\Anaconda\envs\rtdetr\python.exe")
TRAIN_SCRIPT = PROJECT_ROOT / "tools" / "train.py"

CUDA_VISIBLE_DEVICES = "0"
SEED = 4

# ---- 实验定义 ----
EXPERIMENTS = [
    {
        "name": "baseline_assa",
        "config": "configs/rtdetr/baseline_assa.yml",
        "output_dir": r"D:\Learn\RTDETR\RT-DETR-main\output\baseline_assa",
    },
    {
        "name": "fb_c5_assa",
        "config": "configs/rtdetr/fb_c5_assa.yml",
        "output_dir": r"D:\Learn\RTDETR\RT-DETR-main\output\fb_c5_assa",
    },
]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def build_env() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def relaunch_if_needed() -> None:
    current_python = Path(sys.executable).resolve()
    target_python = RTDETR_PYTHON.resolve()

    if current_python != target_python:
        print("当前 Python 不是 rtdetr 环境，正在用 rtdetr 环境重新启动本脚本...")
        import subprocess
        cmd = [str(RTDETR_PYTHON), str(Path(__file__).resolve()), "--relaunched"]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=os.environ.copy())
        raise SystemExit(result.returncode)


def check_required_files() -> None:
    required = [
        (PROJECT_ROOT, "项目根目录"),
        (RTDETR_PYTHON, "rtdetr Python"),
        (TRAIN_SCRIPT, "训练入口 tools/train.py"),
    ]
    for path, desc in required:
        if not path.exists():
            raise FileNotFoundError(f"{desc} 不存在：{path}")
        print(f"[OK] {desc}: {path}")

    for exp in EXPERIMENTS:
        config_file = PROJECT_ROOT / exp["config"]
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在：{config_file}")
        print(f"[OK] {exp['name']} config: {config_file}")

    # verify ASSA support in hybrid_encoder.py
    encoder_path = PROJECT_ROOT / "src" / "zoo" / "rtdetr" / "hybrid_encoder.py"
    encoder_text = encoder_path.read_text(encoding="utf-8", errors="replace")
    if "AdaptiveSparseSelfAttention" not in encoder_text:
        raise RuntimeError("hybrid_encoder.py 中没有检测到 AdaptiveSparseSelfAttention 支持。")
    print("[OK] hybrid_encoder.py 已包含 ASSA 支持")

    # verify FasterBlock support (same check as run_c5.py)
    presnet_path = PROJECT_ROOT / "src" / "nn" / "backbone" / "presnet.py"
    presnet_text = presnet_path.read_text(encoding="utf-8", errors="replace")
    if "class FasterBlock" not in presnet_text or "use_fasterblock" not in presnet_text:
        raise RuntimeError("presnet.py 中没有检测到 FasterBlock 支持。")
    print("[OK] presnet.py 已包含 FasterBlock 支持")


def set_random_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(f"[OK] 随机种子已设置为: {seed}")
    print("[OK] cudnn.benchmark = False")
    print("[OK] cudnn.deterministic = True")


def check_cuda() -> None:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"torch cuda: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，不能进行 GPU 训练。")
    print(f"gpu: {torch.cuda.get_device_name(0)}")


def run_experiment(exp_name: str, config_arg: str, output_dir: Path) -> None:
    """Run a single training experiment via runpy."""
    os.chdir(PROJECT_ROOT)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    console_dir = output_dir / "console_logs"
    console_dir.mkdir(parents=True, exist_ok=True)

    log_path = console_dir / f"{exp_name}_seed{SEED}_{timestamp}.log"
    latest_log_path = console_dir / f"{exp_name}_seed{SEED}_latest.log"

    cmd_text = f"{RTDETR_PYTHON} {TRAIN_SCRIPT} -c {config_arg}"

    with log_path.open("w", encoding="utf-8", errors="replace") as f, \
            latest_log_path.open("w", encoding="utf-8", errors="replace") as latest:
        tee = Tee(sys.stdout, f, latest)
        with redirect_stdout(tee), redirect_stderr(tee):
            header(f"开始训练: {exp_name}")
            print("【控制台日志保存已启用】")
            print(f"运行目录: {PROJECT_ROOT}")
            print(f"配置文件: {config_arg}")
            print(f"输出目录: {output_dir}")
            print(f"随机种子: {SEED}")
            print("等效训练命令:")
            print(cmd_text)
            print("本次完整日志:")
            print(log_path)
            print("最新日志快捷文件:")
            print(latest_log_path)
            print("-" * 90)

            sys.argv = [str(TRAIN_SCRIPT), "-c", config_arg]
            runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")

            header(f"{exp_name} 训练结束")
            print(f"日志已保存: {log_path}")
            print(f"最新日志: {latest_log_path}")


def main() -> None:
    build_env()
    relaunch_if_needed()

    header("检查文件")
    check_required_files()

    header("检查 CUDA")
    check_cuda()

    total = len(EXPERIMENTS)
    for idx, exp in enumerate(EXPERIMENTS, start=1):
        header(f"===== 实验 {idx}/{total}: {exp['name']} =====")
        set_random_seed(SEED)
        try:
            run_experiment(
                exp_name=exp["name"],
                config_arg=exp["config"],
                output_dir=Path(exp["output_dir"]),
            )
        except SystemExit:
            raise
        except Exception:
            print(f"\n实验 {exp['name']} 运行失败，错误信息：")
            traceback.print_exc()
            print(f"\n跳过 {exp['name']}，继续下一个实验...")
            continue

    header("===== 全部实验结束 =====")
    print("实验汇总：")
    for exp in EXPERIMENTS:
        print(f"  {exp['name']}: {exp['output_dir']}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("\n程序运行失败，错误信息如下：")
        traceback.print_exc()
        raise SystemExit(1)
