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

    def _read_struct(self, s: Struct):
        return s.unpack(self._handle.read(s.size))

    @abc.abstractmethod
    def parse(self):
        pass

    def _parse_offset_vector(self):
        offset_vector_size = self._read_struct(VECTOR_SIZE_STRUCT)[0]
        self._offset_vector = np.frombuffer(buffer=self._handle.read(offset_vector_size * data_type_sizes['size_t']),
                                            dtype=SIZE_T_DTYPE)

    def _parse_data_vector(self):
        data_vector_size = self._read_struct(VECTOR_SIZE_STRUCT)[0]
        data_vector_entry_size = data_type_sizes['uint32_t'] + data_type_sizes['float']
        self._data_vector = np.frombuffer(buffer=self._handle.read(data_vector_size * data_vector_entry_size),
                                          dtype=np.dtype([('keys', 'i4'), ('data', 'float32')]))

    def _get_nparray(self) -> np.ndarray:
        """Generate 2d numpy array

        Returns
        -------
        np.ndarray
            dmatrix converted to 2d numpy array
        """
        if self.num_row == 0 or self.num_col == 0:
            return np.empty((self.num_row, self.num_col))
        matrix = np.nan * np.empty((self.num_row * self.num_col))
        sizes = self._offset_vector[1:] - self._offset_vector[:-1]
        add_array = np.repeat(np.arange(0, sizes.size * self.num_col, self.num_col), sizes)
        new_keys = self._data_vector['keys'] + add_array
        matrix[new_keys] = self._data_vector['data']
        return matrix.reshape((self.num_row, self.num_col))
