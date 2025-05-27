from setuptools import setup, find_packages

setup(
    name="point_cloud_tools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "torch", "tqdm", "trimesh", "requests"],
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
