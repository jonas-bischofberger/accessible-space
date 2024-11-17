from setuptools import setup, find_packages

setup(
    name="accessible-space",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "pandas<2.0.0",
        "numpy<1.25.0",
        "scipy",
    ],
    author="Jonas Bischofberger",
    author_email="jonasbischofberger@web.de",
    description="This package implements a refined version of the physical pass model by Spearman et al. (2016) that adds Dangerous Accessible Space (DAS) as a spatial aggregate of reception probabilities.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/jonas-bischofberger/accessible-space",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
#    python_requires='>=3.6',  # Specify the minimum Python version
)
