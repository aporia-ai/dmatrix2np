import unittest
import numpy as np
import xgboost as xgb
from dmatrix2np import dmatrix_to_numpy, InvalidInput
from packaging import version


class TestDmatrix2Numpy(unittest.TestCase):

    def test_none(self):
        with self.assertRaises(InvalidInput):
            dmatrix_to_numpy(None)

    def test_simple_matrix(self):
        ndarr = np.array([[1, 2], [3,4]])
        dmat = xgb.DMatrix(ndarr)
        np.testing.assert_equal(dmatrix_to_numpy(dmat), ndarr)

    def test_empty_matrices(self):
        ndarr = np.empty((0, 0))
        dmat = xgb.DMatrix(ndarr)
        np.testing.assert_equal(dmatrix_to_numpy(dmat), ndarr)

        ndarr = np.empty((1, 0))
        dmat = xgb.DMatrix(ndarr)
        np.testing.assert_equal(dmatrix_to_numpy(dmat), ndarr)

        ndarr = np.empty((0, 1))
        dmat = xgb.DMatrix(ndarr)
        np.testing.assert_equal(dmatrix_to_numpy(dmat), ndarr)

    def test_nan_matrix(self):
        ndarr = np.nan * np.empty((2, 2))
        dmat = xgb.DMatrix(ndarr)
        np.testing.assert_equal(dmatrix_to_numpy(dmat), ndarr)

    def test_large_random_matrix(self):
        ndarr = np.random.rand(1000, 1000).astype(np.float32)
        dmat = xgb.DMatrix(ndarr)
        np.testing.assert_equal(dmatrix_to_numpy(dmat), ndarr)

    def test_nan_row_matrix(self):
        ndarr = np.random.rand(100, 100).astype(np.float32)
        ndarr[10] = np.nan
        dmat = xgb.DMatrix(ndarr)
        np.testing.assert_equal(dmatrix_to_numpy(dmat), ndarr)

    def test_nan_col_matrix(self):
        ndarr = np.random.rand(100, 100).astype(np.float32)
        ndarr = ndarr.T
        ndarr[10] = np.nan
        ndarr = ndarr.T
        dmat = xgb.DMatrix(ndarr)
        np.testing.assert_equal(dmatrix_to_numpy(dmat), ndarr)

    def test_dmatrix_with_labels(self):
        ndarr = np.random.rand(100, 100).astype(np.float32)
        label = np.random.rand(100, 1).astype(np.float32)
        dmat = xgb.DMatrix(ndarr, label=label)
        np.testing.assert_equal(dmatrix_to_numpy(dmat), ndarr)

    def test_dmatrix_with_weight(self):
        ndarr = np.random.rand(100, 100).astype(np.float32)
        weight = np.random.rand(100, 1).astype(np.float32)
        dmat = xgb.DMatrix(ndarr, weight=weight)
        np.testing.assert_equal(dmatrix_to_numpy(dmat), ndarr)

    def test_dmatrix_with_base_margin(self):
        if version.parse(xgb.__version__) < version.parse('1.0.0'):
            return True
        ndarr = np.random.rand(100, 100).astype(np.float32)
        base_margin = np.random.rand(100, 1).astype(np.float32)
        dmat = xgb.DMatrix(ndarr, base_margin=base_margin)
        np.testing.assert_equal(dmatrix_to_numpy(dmat), ndarr)

    def test_complex_dmatrix(self):
        ndarr = np.random.rand(100, 100).astype(np.float32)
        ndarr[10] = np.nan
        ndarr = ndarr.T
        ndarr[10] = np.nan
        ndarr = ndarr.T
        label = np.random.rand(100, 1).astype(np.float32)
        weight = np.random.rand(100, 1).astype(np.float32)
        base_margin = np.random.rand(100, 1).astype(np.float32)
        if version.parse(xgb.__version__) < version.parse('1.0.0'):
            dmat = xgb.DMatrix(ndarr, label=label, weight=weight)
        else:
            dmat = xgb.DMatrix(ndarr, label=label, weight=weight, base_margin=base_margin)
        np.testing.assert_equal(dmatrix_to_numpy(dmat), ndarr)

    def test_simple_vector(self):
        vector = np.array([1, 2])
        with self.assertRaises(ValueError):
            xgb.DMatrix(vector)

    def test_unsupported_shapes(self):
        shapes = [0, (2), (2, 2, 2), (2, 2, 2, 2), (2, 2, 2, 2, 2)]
        for shape in shapes:
            with self.assertRaises(ValueError):
                xgb.DMatrix(np.empty(shape))
