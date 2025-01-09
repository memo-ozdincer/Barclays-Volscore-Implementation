import os
import sys
import platform
import setuptools
from setuptools import Extension
from setuptools import setup
import pybind11

# Adjust this path if needed
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
    version='0.0.1',
    author='You',
    author_email='<your-email>',
    description='Pybind11 wrapper for VolScore',
    ext_modules=ext_modules,
    cmdclass={'build_ext': setuptools.command.build_ext.build_ext},
    zip_safe=False,
)
