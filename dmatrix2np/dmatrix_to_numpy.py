import tempfile
import os
import xgboost as xgb
import numpy as np
from .dmatrix_v_1_0_0_stream_parser import DMatrixStreamParserV1_0_0
from .dmatrix_v_0_80_stream_parser import DMatrixStreamParserV0_80
from .exceptions import InvalidInput
from packaging import version
from contextlib import suppress


def dmatrix_to_numpy(dmatrix: xgb.DMatrix) -> np.ndarray:
    """Convert DMatrix to 2d numpy array

    Parameters
    ----------
    dmatrix : xgb.DMatrix
        DMatrix to convert

    Returns
    -------
    np.ndarray
        2d numpy array with the corresponding DMatrix feature values

    Raises
    ------
    InvalidInput
        Input is not a valid DMatrix
    """
    if not isinstance(dmatrix, xgb.DMatrix):
        raise InvalidInput("Type error: input parameter is not DMatrix")

    stream_parser = DMatrixStreamParserV0_80 if version.parse(xgb.__version__) < version.parse('1.0.0') \
        else DMatrixStreamParserV1_0_0

    # We set delete=False to avoid permissions error. This way, file can be accessed
    # by XGBoost without being deleted while handle is closed
    try:
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            dmatrix.save_binary(fp.name)
            result = stream_parser(fp, dmatrix.num_row(), dmatrix.num_col()).parse()
    finally:
        # We can safely remove the temp file now, parsing process finished
        with suppress(OSError):
            os.remove(fp.name)

    return result
