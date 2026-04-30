from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='OrinFlow_Kernel', 
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name='ofk._C', 
            sources=[
                'src/add/entry.cc',
                'src/add/add.cu'
            ],
            include_dirs=[],
            extra_compile_args={
                'cxx': ['/O2'], 
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_120,code=sm_120', 
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)