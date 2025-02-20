#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages


with open("README_PYPI.rst") as readme_file:
    LONG_DESC = readme_file.read()


requirements = [
    "cvxpy>=1.2.1",
    "dill>=0.3.6",
    "holidays==0.13",
    "holidays-ext==0.0.8",
    "ipython>=7.31.1",
    "matplotlib>=3.4.1",
    "numpy==1.26.0",
    "osqp>=0.6.2",
    "overrides>=2.8.0",
    "pandas>=1.5.0, <2.0.0",
    "patsy>=0.5.2",
    "plotly>=4.12.0",
    "pytest==8.3.4",
    "scipy>=1.15.0, <1.15.2",
    "six>=1.15.0",
    "scikit-learn==1.3.1",
    "statsmodels>=0.13.5",
    "testfixtures>=6.14.2",
    "tqdm>=4.52.0"
]

setup_requirements = []

test_requirements = ["pytest>=3"]

setup(
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
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
    version="1.1.0",
    zip_safe=False,
)
