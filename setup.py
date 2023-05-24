#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages


with open("README_PYPI.rst") as readme_file:
    LONG_DESC = readme_file.read()


requirements = [
    "cvxpy>=1.2.1",
    "dill>=0.3.1.1",
    "holidays-ext>=0.0.7",
    "ipython>=7.31.1",
    "matplotlib>=3.4.1",
    "numpy>=1.22.0",  # support for Python 3.10
    "osqp>=0.6.1",
    "overrides>=2.8.0",
    "pandas>=1.5.0, <2.0.0",
    "patsy>=0.5.1",
    "plotly>=4.12.0",
    "pmdarima>=1.8.0, <=1.8.5",
    "pytest>=4.6.5",
    "pytest-runner>=5.1",
    "scipy>=1.5.4",
    "six>=1.15.0",
    "scikit-learn>=0.24.1",
    "statsmodels>=0.12.2",
    "testfixtures>=6.14.2",
    "tqdm>=4.52.0"]

setup_requirements = ["pytest-runner", ]

test_requirements = ["pytest>=3", ]

setup(
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10"
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
    author="R. Hosseini, A. Chen, K. Yang, S. Patra, Y. Su, R. Arora",
    author_email="reza1317@gmail.com",
    packages=find_packages(include=['greykite', 'greykite.*']),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/linkedin/greykite",
    version="0.5.0",
    zip_safe=False,
)
