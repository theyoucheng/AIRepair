#!/usr/bin/env python

import os

from setuptools import setup, find_packages


# Get a list of all files in the JS directory to include in our module
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "./README.md")) as f:
    README = f.read()

setup(
    name="RNNRepair",
    version="0.0.1",
    description="RNNRepair: Automatic RNN Repair via Model-based Analysis",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/xiaofeixie/rnnrepair",
    author="RNNRepair Team",
    author_email="xiaofei.xfxie@gmail.com",
    license="MIT",
    keywords="api of RNNRepair ",
    packages=find_packages(exclude=['tests', 'tests.*',"*egg-info*",".ptp-sync*"]),
    install_requires=[],
    extras_require={"testing": ["pytest"]},
    python_requires=">=3.5",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Environment :: Web Environment",
        "Framework :: AsyncIO",
        "Intended Audience :: Developers",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: Scientific/Engineering :: GIS",
    ],
)