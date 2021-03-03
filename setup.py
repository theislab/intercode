from pathlib import Path

from setuptools import setup, find_packages

long_description = Path("README.rst").read_text("utf-8")

try:
    from intercode import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ""

setup(
    name="intercode",
    version="0.0.4",
    description="Interpretable Autoencoder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theislab/intercode",
    author=__author__,
    author_email=__email__,
    license="BSD-3-Clause",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        l.strip() for l in Path("requirements.txt").read_text("utf-8").splitlines()
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Environment :: Console",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
)
