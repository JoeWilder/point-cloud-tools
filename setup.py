from setuptools import setup, find_packages
from point_cloud_tools import __version__

setup(
    name="point_cloud_tools",
    version=__version__,
    packages=find_packages(),
    install_requires=["numpy", "torch", "tqdm", "trimesh", "requests", "matplotlib"],
    python_requires=">=3.6",
    author="Joe Wilder",
    description="A Python package for processing point clouds for neural networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JoeWilder/point-cloud-tools",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
