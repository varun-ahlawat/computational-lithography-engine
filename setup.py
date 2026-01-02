"""Setup file for computational lithography engine."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="computational-lithography-engine",
    version="0.1.0",
    author="VARUN",
    description="A differentiable physics engine for computational lithography",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yup-VARUN/computational-lithography-engine",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
)
