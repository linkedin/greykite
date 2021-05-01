#!/usr/bin/env python

"""The setup script."""

import sys
from setuptools import setup, find_packages

with open("README.rst") as readme_file:
   LONG_DESC = readme_file.read()

requirements = [
    "cvxpy==1.1.4",
    "fbprophet==0.5",
    "holidays==0.9.10",  # 0.10.2,
    "ipykernel==4.8.2",
    "ipython==7.1.1",
    "ipywidgets==7.2.1",
    "jupyter==1.0.0",
    "jupyter-client==5.2.3",
    "jupyter-console==6.",  # used version 6 to avoid conflict with ipython version
    "jupyter-core==4.4.0",
    "matplotlib==3.0.3",
    "nbformat==4.4.0",
    "notebook==5.4.1",
    "numpy==1.20.1",
    "osqp==0.6.1",
    "overrides==2.8.0",
    "pandas==1.1.3",
    "patsy==0.5.1",
    "Pillow==8.0.1",
    "plotly==3.10.0",
    "pystan==2.18.0.0",
    "pyzmq==17.1.2",
    "scipy==1.5.4",
    "seaborn==0.9.0",
    "six==1.15.0",
    "scikit-learn==0.24.1",
    "Sphinx==3.2.1",
    "sphinx-gallery==0.6.1",
    "sphinx-rtd-theme==0.4.2",
    "statsmodels==0.12.0",
    "testfixtures==6.14.2",
    "tqdm==4.52.0"]

# Here we change some dependencies versions according to python version
# to minimize the chance of install failures. 
if sys.version_info < (3 , 7):
    requirements = ["numpy==1.19.1" if i=="numpy==1.20.1" else i for i in requirements]

setup_requirements = ["pytest-runner", ]

test_requirements = ["pytest>=3", ]

setup(
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
    ],
    description="A python package for flexible forecasting",
    long_description=LONG_DESC,
    entry_points={
        "console_scripts": [
            "greykite=greykite.cli:main",
        ],
    },
    install_requires=requirements,
    license="BSD-2-CLAUSE",
    include_package_data=True,
    keywords="greykite",
    name="greykite",
    author="R. Hosseini, A Chen, K. Yang, S. Patra, R. Arora",
    author_email="reza1317@gmail.com",
    packages=find_packages(include=['greykite', 'greykite.*']),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/linkedin/greykite",
    version="0.0.1",
    zip_safe=False,
)
