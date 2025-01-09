import os
import sys
import setuptools
from setuptools import Extension, setup
import pybind11

# Adjust paths as needed
inc = pybind11.get_include()

ext_modules = [
    Extension(
        'volscore_wrapper',
        sources=[
            'volscore_wrapper/volscore_pybind.cpp',
            '../cpp/src/volscore.cpp'
        ],
        include_dirs=[
            inc,
            '../cpp/include'
        ],
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
]

setup(
    name='volscore_wrapper',
    version='1.0.0',
    description='Advanced C++ VolScore library for measuring volatility, skew, kurtosis & bridging to Python w/ pybind11',
    author='You',
    author_email='you@example.com',
    ext_modules=ext_modules,
    cmdclass={'build_ext': setuptools.command.build_ext.build_ext},
    zip_safe=False,
    packages=['volscore_wrapper'],
)