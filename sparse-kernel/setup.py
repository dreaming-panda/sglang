from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='block_sparse_attention',
    ext_modules=[
        CUDAExtension(
            name='block_sparse_attention',
            sources=['block_sparse_attention.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--expt-relaxed-constexpr', '-gencode=arch=compute_89,code=sm_89']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)