import os
import shutil
import subprocess
import sys
import torch

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        project_root = os.path.dirname(os.path.abspath(__file__))
        build_temp_dir = os.path.join(project_root, "build")
        build_lib_dir = os.path.join(project_root, "ofk")

        os.makedirs(build_lib_dir, exist_ok=True)
        os.makedirs(build_temp_dir, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_temp_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}", 
            "-DCMAKE_BUILD_TYPE=Release"
        ]

        num_jobs = os.cpu_count() or 4

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp_dir)
        subprocess.check_call(["cmake", "--build", ".", "--config", "Release", f"-j{num_jobs}"], cwd=build_temp_dir)

        ext_suffix = ".pyd" if os.name == "nt" else ".so"
        so_filename = f"orinflow_kernel{ext_suffix}" 
        
        so_src_path = os.path.join(build_temp_dir, so_filename)
        if not os.path.exists(so_src_path):
            so_src_path_msvc = os.path.join(build_temp_dir, "Release", so_filename)
            if os.path.exists(so_src_path_msvc):
                so_src_path = so_src_path_msvc
        so_dst_path = os.path.join(build_lib_dir, so_filename)
        if os.path.exists(so_src_path):
            os.makedirs(os.path.dirname(so_dst_path), exist_ok=True)
            shutil.copy(so_src_path, so_dst_path)
            print(f"成功将动态库搬运至: {so_dst_path}") 
        else:
            raise FileNotFoundError(f"编译成功，但未能找到预期的动态库文件: {so_src_path}")

setup(
    name="orinflow_kernel",
    version="0.1.0",
    description="Custom PyTorch Operators via CMake",
    packages=["ofk"],
    ext_modules=[CMakeExtension("ofk.orinflow_kernel")],
    cmdclass={"build_ext": CMakeBuild},
    package_data={"ofk": ["*.so", "*.pyd"]},
    install_requires=["torch"],
)
