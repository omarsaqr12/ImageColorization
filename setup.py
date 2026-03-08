"""
Setup script for Image Colorization with Knowledge Distillation
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="image-colorization-distillation",
    version="1.0.0",
    author="Omar Mokhtar Gaber Saqr",
    author_email="omar_saqr@aucegypt.edu",
    description="Automatic image colorization using deep learning with knowledge distillation for model compression",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/image-colorization-distillation",
    packages=find_packages(where="code"),
    package_dir={"": "code"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "colorize-demo=colorization.demo_release:main",
        ],
    },
)
