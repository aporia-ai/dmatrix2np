import tempfile
import os
import xgboost as xgb
import numpy as np
from .dmatrix_v_1_0_2_stream_parser import DMatrixStreamParserV1_0_2


def dmatrix_to_numpy(dmatrix):
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        dmatrix.save_binary(fp.name)
        result = DMatrixStreamParserV1_0_2(fp, dmatrix.num_row(), dmatrix.num_col()).get_nparray()
    os.remove(fp.name)
    return result

data = np.array([[1,2,3], [None, None, 4], [5,6,7], [8,9,0]])
labels = np.array([1,0,1])
d_mat = xgb.DMatrix(data, labels)
d_mat.save_binary(r'C:\Temp\just_try.bin')
# print(dmatrix_to_numpy(d_mat))
