import os
import io

from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()

setup(
    name="gen",
    version='1.0',
    packages=find_packages("."),
    package_dir={"": "."},
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "torch>=1.9",
    ],
    include_package_data=True,
    license="MIT",
)
