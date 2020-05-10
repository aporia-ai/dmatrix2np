from .dmatrix_stream_parser import DMatrixStreamParser
from .exceptions import InvalidStructure
from .common import FieldDataType, size_t_size, size_t_dtype_str, data_type_sizes, byte_order_str
import numpy as np


class DMatrixStreamParserV0_90(DMatrixStreamParser):

    kMagic = 0xffffab01
    kVersion = 2

    def __init__(self, buffer_reader, num_row, num_col):
        self._handle = buffer_reader
        self.num_row = num_row
        self.num_col = num_col
        self._parse()

    def _parse(self):
        self._handle.seek(0)
        self._parse_magic()
        self._parse_version()
        self._skip_fields()
        self._parse_offset_vector()
        self._parse_data_vector()
        self._to_nparray()

    def _parse_magic(self):
        kMagic = self._handle.read(4)
        if kMagic != self.kMagic.to_bytes(4, byteorder=byte_order_str):
            raise InvalidStructure('Invalid magic')

    def _parse_version(self):
        version = self._handle.read(4)
        if version != self.kVersion.to_bytes(4, byteorder=byte_order_str):
            raise InvalidStructure('Invalid version')

    def _skip_fields(self):
        # Skip num_row_, num_col_, num_nonzero_ (all uint64_t)
        self._handle.read(24)

        # skip 5 info vectors
        self._num_of_fields = int.from_bytes(self._handle.read(8), byte_order_str)
        self._fields_offset = self._handle.tell()
        for _ in range(self._num_of_fields):
            self._skip_field()

    def _skip_vector(self):
        # Skip field name (pascal string)
        name_size = int.from_bytes(self._handle.read(8), byte_order_str)
        self._handle.read(name_size)

        # Find field type
        field_type = FieldDataType(int.from_bytes(self._handle.read(1), byte_order_str))
        is_scalar = self._handle.read(1) == b'\x01'

        if is_scalar:
            self._handle.read(data_type_sizes[field_type.name])
        else:
            # Skip shape.first, shape.second
            self._handle.read(16)

            vector_size = int.from_bytes(self._handle.read(8), byte_order_str)

            # Skip vector
            self._handle.read(vector_size * data_type_sizes[field_type.name])

    def _parse_offset_vector(self):
        self._offset_vector_offset = self._handle.tell()
        self._offset_vector_size = int.from_bytes(self._handle.read(size_t_size), byte_order_str)
        self._offset_vector = np.frombuffer(buffer=self._handle.read(self._offset_vector_size * size_t_size),
                                            dtype=size_t_dtype_str)

    def _parse_data_vector(self):
        self._data_vector_offset = self._handle.tell()
        self._data_vector_size = int.from_bytes(self._handle.read(size_t_size), byte_order_str)
        self._data_vector = np.frombuffer(buffer=self._handle.read(self._data_vector_size * 8),
                                          dtype=np.dtype([('keys', 'i4'), ('data', 'f4')]))

    def _to_nparray(self):
        matrix = np.nan * np.empty((self.num_row * self.num_col))
        sizes = self._offset_vector[1:] - self._offset_vector[:-1]
        add_array = np.repeat(np.arange(0, sizes.size * self.num_col, self.num_col), sizes)
        new_keys = self._data_vector['keys'] + add_array
        matrix[new_keys] = self._data_vector['data']
        self.np_array = matrix.reshape((self.num_row, self.num_col))

    def get_nparray(self):
        return self.np_array
