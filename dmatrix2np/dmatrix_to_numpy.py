import tempfile
import os
import xgboost as xgb
import numpy as np
from .dmatrix_v_1_0_0_stream_parser import DMatrixStreamParserV1_0_0
from .dmatrix_v_0_80_stream_parser import DMatrixStreamParserV0_80
from .exceptions import UnsupportedVersion, InvalidInput

XGBOOST_VER_2_STREAM_PARSER = {
    '0.80': DMatrixStreamParserV0_80,
    '0.81': DMatrixStreamParserV0_80,
    '0.90': DMatrixStreamParserV0_80,
    '1.0.0rc1': DMatrixStreamParserV1_0_0,
    '1.0.0rc2': DMatrixStreamParserV1_0_0,
    '1.0.0': DMatrixStreamParserV1_0_0,
    '1.0.1': DMatrixStreamParserV1_0_0,
    '1.0.2': DMatrixStreamParserV1_0_0,
}

def dmatrix_to_numpy(dmatrix):
    if not isinstance(dmatrix, xgb.DMatrix):
        raise InvalidInput("Type error: input parameter is not DMatrix")

    xgb_version = xgb.__version__
    if xgb_version not in XGBOOST_VER_2_STREAM_PARSER:
        raise UnsupportedVersion
    stream_parser = XGBOOST_VER_2_STREAM_PARSER[xgb_version]

    # We set delete=False to avoid permissions error. This way, file can be accessed
    # by XGBoost without being deleted while handle is closed
    fp = tempfile.NamedTemporaryFile(delete=False)
    try:
        dmatrix.save_binary(fp.name)
        result = stream_parser(fp, dmatrix.num_row(), dmatrix.num_col()).parse()
    finally:
        fp.close()
        os.remove(fp.name)
    return result
