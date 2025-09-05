from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name="rtopk_cuda",
    ext_modules=[
        CUDAExtension(
            name="rtopk_cuda",
            sources=[
                "rtopk_binding.cpp",
                "rtopk_ops.cu",
                "rtopk_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)


