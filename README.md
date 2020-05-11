# dmatrix2np

[![Tests](https://github.com/aporia-ai/dmatrix2np/workflows/Test/badge.svg)](https://github.com/aporia-ai/dmatrix2np/actions?workflow=Test) [![PyPI](https://img.shields.io/pypi/v/dmatrix2np.svg)](https://pypi.org/project/dmatrix2np/)

Convert XGBoost's DMatrix format to np.array.

## Usage

To install the library, run:

    pip install dmatrix2np

Then, you can call in your code:

    from dmatrix2np import dmatrix2np

    converted_np_array = dmatrix_to_numpy(dmatrix)

## Development

We use [poetry](https://python-poetry.org/) for development:

    pip install poetry

To install all dependencies and run tests:

    poetry run pytest
    
To run tests on the entire matrix (Python 3.6, 3.7, 3.8 + XGBoost 0.80, 0.90, 1.0):
    
    pip install tox
    tox
