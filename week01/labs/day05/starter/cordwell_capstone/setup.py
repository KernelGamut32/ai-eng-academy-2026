"""Minimal setuptools config so `pip install -e .` makes `cordwell` importable.

Deliberately plain: no pyproject.toml, no build backend config. Run:
    pip install -e .
"""
from setuptools import find_packages, setup

setup(
    name="cordwell",
    version="0.1.0",
    description="Week 1 capstone: Cordwell Home & Hardware data-cleaning pipeline",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=["pandas>=3.0", "numpy>=2.0", "pyarrow>=17.0", "pandera>=0.20"],
)
