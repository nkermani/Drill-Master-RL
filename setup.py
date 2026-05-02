# setup.py - Easy installation for Drill-Master-RL

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="drill-master-rl",
    version="0.1.0",
    author="Nathan Kermani",
    author_email="kermani.nathan@gmail.com",
    description="Reinforcement Learning for Multi-Robot Fleet Navigation under Uncertainty",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nkermani/Drill-Master-RL",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "torch": [
            "torch>=2.0.0",
            "torch-geometric>=2.3.0",
        ],
        "dev": [
            "pytest>=7.3.0",
            "black>=23.0.0",
            "jupyter>=1.0.0",
            "tqdm>=4.65.0",
        ],
    },
)
