"""Setuptools entry point for the DDPM project."""

from setuptools import find_packages, setup


INSTALL_REQUIRES = [
    "torch>=2.9.0",
    "torchvision>=0.24.0",
    "numpy>=2.2.0",
    "matplotlib>=3.10.0",
    "tqdm>=4.67.0",
    "PyYAML>=6.0.2",
    "scipy>=1.15.0",
    "Pillow>=11.1.0",
]


setup(
    name="ddpm-from-scratch",
    version="0.1.0",
    description="A PyTorch implementation of DDPM image generation.",
    packages=find_packages(),
    python_requires=">=3.10,<3.15",
    install_requires=INSTALL_REQUIRES,
)
