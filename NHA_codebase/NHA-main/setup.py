# -*- coding: utf-8 -*-

import ast
import os
import re
from pathlib import Path

from setuptools import find_packages, setup


setup(
    name='nha_fla',
    description='Implementations of Native Hybrid Attention',
    author='Jusen Du',
    author_email='dujusen@gmail.com',
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.5',
        'transformers>=4.45.0',
        'datasets>=3.3.0',
        'einops',
        'ninja',
        'accelerate',
        'deepspeed',
        'protobuf',
        'sentencepiece',
        'flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention'
    ],
    extras_require={
        'conv1d': ['causal-conv1d>=1.4.0']
    }
)
