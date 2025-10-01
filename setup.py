#!/usr/bin/env python3
"""
Setup script for Humigence CLI
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="humigence",
    version="1.0.0",
    author="Humigence Team",
    author_email="contact@humigence.ai",
    description="Your AI. Your pipeline. Zero code. - Complete MLOps toolkit for LLM fine-tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/humigence",
    project_urls={
        "Bug Reports": "https://github.com/your-username/humigence/issues",
        "Source": "https://github.com/your-username/humigence",
        "Documentation": "https://github.com/your-username/humigence/wiki",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: User Interfaces",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "humigence=cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
    },
    keywords=[
        "machine-learning",
        "fine-tuning",
        "lora",
        "qlora",
        "llm",
        "mlops",
        "cli",
        "wizard",
        "gpu",
        "unsloth",
        "transformers",
        "pytorch",
    ],
    zip_safe=False,
)
