# The MIT License (MIT)
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import sys
from setuptools import setup, find_packages, dist
import glob
import logging

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

PACKAGE_NAME = 'neuralff'
DESCRIPTION = 'research on neural fluid fields'
URL = 'https://github.com/tovacinni/neural-fluid-fields'
AUTHOR = 'Towaki Takikawa, Mohammad Mozaffari'
LICENSE = 'MIT License'
version = '0.1.0'

def get_extensions():
    extra_compile_args = {'cxx': ['-O3']} 
    define_macros = []
    include_dirs = []
    extensions = []
    sources = glob.glob('neuralff/csrc/**/*.cpp', recursive=True)
 
    if len(sources) == 0:
        print("No source files found for extension, skipping extension compilation")
        return None

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [("WITH_CUDA", None), ("THRUST_IGNORE_CUB_VERSION_CHECK", None)]
        sources += glob.glob('neuralff/csrc/**/*.cu', recursive=True)
        extension = CUDAExtension
        extra_compile_args.update({'nvcc': ['-O3']})
        #include_dirs = get_include_dirs()
    else:
        assert(False, "CUDA is not available. Set FORCE_CUDA=1 for Docker builds")

    extensions.append(
        extension(
            name='neuralff._C',
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            #include_dirs=include_dirs
        )
    )

    for ext in extensions:
        ext.libraries = ['cudart_static' if x == 'cudart' else x
                         for x in ext.libraries]
 
    return extensions

if __name__ == '__main__':
    setup(
        # Metadata
        name=PACKAGE_NAME,
        version=version,
        author=AUTHOR,
        description=DESCRIPTION,
        url=URL,
        license=LICENSE,
        python_requires='~=3.8',

        # Package info
        packages=['neuralff'],
        include_package_data=True,
        zip_safe=True,
        ext_modules=get_extensions(),
        cmdclass={
            'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)    
        }

    )
