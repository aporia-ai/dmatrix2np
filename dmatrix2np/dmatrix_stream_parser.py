import abc
import numpy as np


class DMatrixStreamParser(metaclass=abc.ABCMeta):
    """Abstract base class for DMatrix stream parser."""
    @abc.abstractmethod
    def get_nparray(self) -> np.ndarray:
        """Generate 2d numpy array

        Returns
        -------
        np.ndarray
            dmatrix converted to 2d numpy array
        """
        return
