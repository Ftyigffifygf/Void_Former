from setuptools import setup, find_packages

setup(
    name="voidformer",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "harness-plugins*"]),
)
