import os
import shutil
import subprocess
import sys
import torch

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        # name 定义了最终在 Python 中存在的模块路径
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        project_root = os.path.dirname(os.path.abspath(__file__))
        # cmake 中间产物固定放在项目内 _cmake_build/，不随调用方式漂移
        build_temp_dir = os.path.join(project_root, "_cmake_build")
        # .pyd/.so 始终落到 ofk/ 源码目录
        build_lib_dir = os.path.join(project_root, "ofk")

        os.makedirs(build_lib_dir, exist_ok=True)
        os.makedirs(build_temp_dir, exist_ok=True)

        # 核心配置：告诉 CMake 将编译过程放在临时车间，并指定当前环境的 Python 路径
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_temp_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}", 
            "-DCMAKE_BUILD_TYPE=Release"
        ]

        # 动态获取 CPU 核心数，避免写死导致在低配机器上编译时内存溢出 (OOM)
        num_jobs = os.cpu_count() or 4

        # 阶段一：CMake 生成构建系统
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp_dir)
        
        # 阶段二：执行并行编译
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release", f"-j{num_jobs}"], 
            cwd=build_temp_dir
        )

        # 阶段三：手动精准搬运 (关键收尾点)
        ext_suffix = ".pyd" if os.name == "nt" else ".so"
        so_filename = f"orinflow_kernel{ext_suffix}" 
        
        # 默认按照 Linux/单配置生成器的路径寻找
        so_src_path = os.path.join(build_temp_dir, so_filename)
        
        # 兜底策略：如果是 Windows MSVC，文件会被强行塞进 Release 子目录
        if not os.path.exists(so_src_path):
            so_src_path_msvc = os.path.join(build_temp_dir, "Release", so_filename)
            if os.path.exists(so_src_path_msvc):
                so_src_path = so_src_path_msvc
        
        # 将动态库强制拷贝到纯 Python 包 ofk 目录下
        so_dst_path = os.path.join(build_lib_dir, so_filename)

        if os.path.exists(so_src_path):
            os.makedirs(os.path.dirname(so_dst_path), exist_ok=True)
            shutil.copy(so_src_path, so_dst_path)
            print(f"成功将动态库搬运至: {so_dst_path}") # 加一句打印，让你装的时候心里有底
        else:
            raise FileNotFoundError(f"编译成功，但未能找到预期的动态库文件: {so_src_path}")

setup(
    name="orinflow_kernel",
    version="0.1.0",
    description="Custom PyTorch Operators via CMake",
    # 告诉 setup 纯 Python 代码所在的文件夹
    packages=["ofk"],
    # 定义 CMake 扩展，将其挂载到 ofk 命名空间下
    ext_modules=[CMakeExtension("ofk.orinflow_kernel")],
    cmdclass={"build_ext": CMakeBuild},
    # 确保打包部署时，包含 ofk 目录下的所有动态库文件
    package_data={"ofk": ["*.so", "*.pyd"]},
    install_requires=["torch"],
)
