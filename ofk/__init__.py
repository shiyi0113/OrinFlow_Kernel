import os
import torch
from pathlib import Path

current_dir = Path(__file__).parent

# 同时收集 .pyd (Windows) 和 .so (Linux/macOS)
lib_files = list(current_dir.glob("orinflow_kernel*.pyd")) + \
            list(current_dir.glob("orinflow_kernel*.so"))

if not lib_files:
    raise FileNotFoundError(
        f"在 {current_dir} 目录下未找到已编译的底层算子库 (.pyd 或 .so)。\n"
        "请确保您已经正确执行了 C++/CUDA 的编译与安装步骤。"
    )

lib_path = str(lib_files[0])
torch.ops.load_library(lib_path)
