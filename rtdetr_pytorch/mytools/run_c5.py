# -*- coding: utf-8 -*-
r"""
FB-C5 单实验训练脚本（固定随机种子 seed=4）

放置位置建议：
D:\Learn\RTDETR\RT-DETR-main\rtdetr_pytorch\mytools\run_c5.py

运行方式：
VS Code 右键本文件 -> Run Python File in Terminal

功能：
1. 只运行 configs/rtdetr/fb_c5.yml
2. 固定随机种子为 4
3. 保存完整控制台输出到 output/m1_fasterblock/console_logs
4. 不自动关机
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
CONFIG_ARG = "configs/rtdetr/fb_c5.yml"
CONFIG_FILE = PROJECT_ROOT / CONFIG_ARG
PRESNET_FILE = PROJECT_ROOT / "src" / "nn" / "backbone" / "presnet.py"
OUTPUT_DIR = Path(r"D:\Learn\RTDETR\RT-DETR-main\output\fb_c5")
CONSOLE_LOG_DIR = OUTPUT_DIR / "console_logs"

CUDA_VISIBLE_DEVICES = "0"
SEED = 4


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
        (CONFIG_FILE, "M1-FasterBlock 配置文件"),
        (PRESNET_FILE, "PResNet 源码"),
    ]
    for path, desc in required:
        if not path.exists():
            raise FileNotFoundError(f"{desc} 不存在：{path}")
        print(f"[OK] {desc}: {path}")

    presnet_text = PRESNET_FILE.read_text(encoding="utf-8", errors="replace")
    if "class FasterBlock" not in presnet_text or "use_fasterblock" not in presnet_text:
        raise RuntimeError(
            "当前 presnet.py 中没有检测到 FasterBlock 支持。\n"
            "请先用 presnet_fasterblock.py 替换 src\\nn\\backbone\\presnet.py。"
        )
    print("[OK] presnet.py 已包含 FasterBlock 支持")

    CONSOLE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] 输出目录: {OUTPUT_DIR}")
    print(f"[OK] 控制台日志目录: {CONSOLE_LOG_DIR}")


def set_random_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic settings improve reproducibility. Some CUDA ops may still have minor nondeterminism.
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


def run_train_with_seed() -> None:
    os.chdir(PROJECT_ROOT)
    build_env()
    set_random_seed(SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = CONSOLE_LOG_DIR / f"fb_c5_seed{SEED}_{timestamp}.log"
    latest_log_path = CONSOLE_LOG_DIR / f"fb_c5_seed{SEED}_latest.log"

    cmd_text = f"{RTDETR_PYTHON} {TRAIN_SCRIPT} -c {CONFIG_ARG}"

    with log_path.open("w", encoding="utf-8", errors="replace") as f, \
            latest_log_path.open("w", encoding="utf-8", errors="replace") as latest:
        tee = Tee(sys.stdout, f, latest)
        with redirect_stdout(tee), redirect_stderr(tee):
            header("开始训练 FB-C5")
            print("【控制台日志保存已启用】")
            print(f"运行目录: {PROJECT_ROOT}")
            print(f"配置文件: {CONFIG_FILE}")
            print(f"输出目录: {OUTPUT_DIR}")
            print(f"随机种子: {SEED}")
            print("等效训练命令:")
            print(cmd_text)
            print("本次完整日志:")
            print(log_path)
            print("最新日志快捷文件:")
            print(latest_log_path)
            print("-" * 90)

            # Run tools/train.py inside this process so that the seed set above is active.
            sys.argv = [str(TRAIN_SCRIPT), "-c", CONFIG_ARG]
            runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")

            header("FB-C5 训练结束")
            print(f"日志已保存: {log_path}")
            print(f"最新日志: {latest_log_path}")


def main() -> None:
    build_env()
    relaunch_if_needed()

    header("检查文件")
    check_required_files()

    header("检查 CUDA")
    check_cuda()

    run_train_with_seed()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("\n程序运行失败，错误信息如下：")
        traceback.print_exc()
        raise SystemExit(1)
