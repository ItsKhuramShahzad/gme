"""
setup.py — hubbard_gme package
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name="hubbard_gme",
    version="0.3.0",
    author="Khuram Shahzad",
    description=(
        "Genuine Multipartite Entanglement analysis for the Hubbard model "
        "using n-body Reduced Density Matrices"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "hubbard-gme = hubbard_gme.pipeline:run_full_GME_pipeline",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
    ],
)
