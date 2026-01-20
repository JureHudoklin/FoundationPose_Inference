from setuptools import setup, find_packages
import os

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="foundation_pose",
    version="0.1.0",
    description="FoundationPose",
    author="NVIDIA CORPORATION",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
