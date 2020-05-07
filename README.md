# dmatrix2np

[![Tests](https://github.com/aporia-ai/dmatrix2np/workflows/Test/badge.svg)](https://github.com/aporia-ai/dmatrix2np/actions?workflow=Test) [![PyPI](https://img.shields.io/pypi/v/dmatrix2np.svg)](https://pypi.org/project/dmatrix2np/)

## Usage

To install the library, run:

    pip install dmatrix2np

Then, you can call in your code:

    from dmatrix2np import dmatrix2np
    converted_np_array = dmatrix2np(dmatrix)

## Development

We use [poetry](https://python-poetry.org/) and [tox](https://tox.readthedocs.io/en/latest/) for development:

    pip install poetry tox

To install all dependencies and run tests:

    tox

Our test matrix includes Python versions 3.6, 3.7, 3.8, and XGBoost versions 0.80, 0.90, 1.0.
