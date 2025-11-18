#!/usr/bin/env python

from setuptools import setup, find_packages

if __name__ == '__main__':

    setup(
        name='unduliner',
        version='1.0',
        author='Halimat Chisom Atanda',
        author_email='gearthdexter@gmail.com',
        description=("CLI tool to link variant to methylation change from nanopore sequence data"),
        license='MIT',
        url='git clone https://github.com/Chisomgold/unduliner.git',
        scripts=['unduliner'],
        packages=find_packages(),
        install_requires = [
            'torch',
            'torch-summary',
            'torchvision',
            'matplotlib',
            'seaborn',
            'pysam',
            'bx-python',
            'scikit-learn',
            'scipy',
            'numpy',
            'pillow'
        ]

    )
