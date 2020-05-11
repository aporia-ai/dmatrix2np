import tempfile
import os
import xgboost as xgb
from .dmatrix_v_1_0_0_stream_parser import DMatrixStreamParserV1_0_0
from .dmatrix_v_0_80_stream_parser import DMatrixStreamParserV0_80
from .exceptions import UnsupportedVersion, InvalidInput
from packaging import version


def dmatrix_to_numpy(dmatrix):
    if not isinstance(dmatrix, xgb.DMatrix):
        raise InvalidInput("Type error: input parameter is not DMatrix")

    stream_parser = DMatrixStreamParserV0_80 if version.parse(xgb.__version__) < version.parse('1.0.0') \
        else DMatrixStreamParserV1_0_0

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
