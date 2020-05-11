import abc
import numpy as np
from struct import Struct
from .common import SIZE_T_DTYPE, data_type_sizes, VECTOR_SIZE_STRUCT


class DMatrixStreamParser(metaclass=abc.ABCMeta):
    """Abstract base class for DMatrix stream parser."""

    def __init__(self, buffer_reader, num_row, num_col):
        self._handle = buffer_reader
        self.num_row = num_row
        self.num_col = num_col
        self._offset_vector = []
        self._data_vector = {}

    @abc.abstractmethod
    def parse(self) -> np.ndarray:
        """Parse DMatrix to numpy 2d array

        Returns
        -------
        np.ndarray
            DMatrix values in numpy 2d array
        """
        pass

    def _read_struct(self, s: Struct):
        return s.unpack(self._handle.read(s.size))

    def _parse_offset_vector(self):
        offset_vector_size, = self._read_struct(VECTOR_SIZE_STRUCT)
        self._offset_vector = np.frombuffer(buffer=self._handle.read(offset_vector_size * data_type_sizes['size_t']),
                                            dtype=SIZE_T_DTYPE)

    def _parse_data_vector(self):
        data_vector_size, = self._read_struct(VECTOR_SIZE_STRUCT)
        data_vector_entry_size = data_type_sizes['uint32_t'] + data_type_sizes['float']
        self._data_vector = np.frombuffer(buffer=self._handle.read(data_vector_size * data_vector_entry_size),
                                          dtype=np.dtype([('column_index', 'i4'), ('data', 'float32')]))

    def _get_nparray(self) -> np.ndarray:
        """Generate 2d numpy array

        Returns
        -------
        np.ndarray
            dmatrix converted to 2d numpy array
        """
        # When the matrix is flat, there are no values and matrix could be generated immediately
        if self.num_row == 0 or self.num_col == 0:
            return np.empty((self.num_row, self.num_col))

        # Create flat matrix filled with nan values
        matrix = np.nan * np.empty((self.num_row * self.num_col))

        # The offset vector contains the offsets of the values in the data vector for each row according to its index
        # We create size vector that contains the size of each row according to its index
        size_vector = self._offset_vector[1:] - self._offset_vector[:-1]

        # Since we work with flat matrix we want to convert 2d index (x, y) to 1d index (x*num_col + y)
        # The data vector only keep column index and data,
        # increase vector is the addition needed for the column index to be 1d index
        increase_vector = np.repeat(np.arange(0, size_vector.size * self.num_col, self.num_col), size_vector)
        flat_indexes = self._data_vector['column_index'] + increase_vector

        # Values assignment
        matrix[flat_indexes] = self._data_vector['data']
        return matrix.reshape((self.num_row, self.num_col))
