# dmatrix2np

[![Tests](https://github.com/aporia-ai/dmatrix2np/workflows/Test/badge.svg)](https://github.com/aporia-ai/dmatrix2np/actions?workflow=Test) [![PyPI](https://img.shields.io/pypi/v/dmatrix2np.svg)](https://pypi.org/project/dmatrix2np/)

Convert XGBoost's DMatrix format to np.array.

<p align="center">Made with :heart: by <a href="https://aporia.com" target="_blank">Aporia</a></p>

## Usage

To install the library, run:

    pip install dmatrix2np

Then, you can call in your code:

    from dmatrix2np import dmatrix_to_numpy

    converted_np_array = dmatrix_to_numpy(dmatrix)

## Development

We use [poetry](https://python-poetry.org/) for development:

    pip install poetry

To install all dependencies and run tests:

    poetry run pytest
    
To run tests on the entire matrix (Python 3.6, 3.7, 3.8, 3.9 + XGBoost 0.80, 0.90, 1.0, 1.1, 1.2, 1.3, 1.4):
    
    pip install tox
    tox

### The Research Behind

In the flowing blog post we describe the research that has been done to understand the DMatrix format and creating this library:
[DMatrix to Numpy Research](https://www.aporia.com/how-to-convert-dmatrix-to-numpy-format-for-your-machine-learning-model-3/)
