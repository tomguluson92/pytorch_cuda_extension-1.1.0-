# -*- coding: utf-8 -*-
"""
    @date: 2020.02.19
    @author: samuel ko
    @readme: target of this repo is help you to fulfill a multi-cuda extension in one setup.py.
"""
from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []

ext_modules = [
    CUDAExtension('add_one_cuda', [
        'cuda_ext/test1_cuda.cpp',
        'cuda_ext/test1_cuda_kernel.cu',
        ])
    ]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image']

# https://pytorch.org/docs/master/cpp_extension.html
setup(
    description='PyTorch implementation of <your own cuda extension>',
    author='samuel ko',
    author_email='samuel.gao023@gmail.com',
    license='MIT License',
    version='1.3.0',
    name='add_one',
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)