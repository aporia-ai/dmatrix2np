from .dmatrix_stream_parser import DMatrixStreamParser
from .exceptions import InvalidStructure
from .common import data_type_sizes, SIZE_T_DTYPE, BYTE_ORDER_STR
import numpy as np


class DMatrixStreamParserV0_80(DMatrixStreamParser):

    NUM_OF_SCALAR_FIELDS = 3
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
        kMagic = self._handle.read(data_type_sizes['int'])
        if kMagic != self.kMagic.to_bytes(data_type_sizes['int'], byteorder=BYTE_ORDER_STR):
            raise InvalidStructure('Invalid magic')

    def _parse_version(self):
        version = self._handle.read(data_type_sizes['int32_t'])
        if version != self.kVersion.to_bytes(data_type_sizes['int32_t'], byteorder=BYTE_ORDER_STR):
            raise InvalidStructure('Invalid version')

    def _skip_fields(self):
        # Skip num_row_, num_col_, num_nonzero_ (all uint64_t)
        self._handle.read(self.NUM_OF_SCALAR_FIELDS * data_type_sizes['uint64_t'])

        # skip info's vector fields (labels_, group_ptr_, qids_, weights_, root_index_, base_margin_)
        vectors_entry_sizes = [
            data_type_sizes['float'],  # labels_
            data_type_sizes['uint32_t'],  # group_ptr_
            data_type_sizes['uint64_t'],  # qids_
            data_type_sizes['float'],  # weights_
            data_type_sizes['uint32_t'],  # root_index_
            data_type_sizes['float'],  # base_margin_
        ]

        # Each vector field starts with uint64_t size indicator
        # followed by number of vector entries equals to indicated size
        for vector_entry_size in vectors_entry_sizes:
            vector_size = int.from_bytes(self._handle.read(data_type_sizes['uint64_t']), BYTE_ORDER_STR)
            self._handle.read(vector_size * vector_entry_size)

    def _parse_offset_vector(self):
        offset_vector_size = int.from_bytes(self._handle.read(data_type_sizes['uint64_t']), BYTE_ORDER_STR)
        self._offset_vector = np.frombuffer(buffer=self._handle.read(offset_vector_size * data_type_sizes['size_t']),
                                            dtype=SIZE_T_DTYPE)

    def _parse_data_vector(self):
        data_vector_size = int.from_bytes(self._handle.read(data_type_sizes['uint64_t']), BYTE_ORDER_STR)
        data_vector_entry_size = data_type_sizes['uint32_t'] + data_type_sizes['float']
        self._data_vector = np.frombuffer(buffer=self._handle.read(data_vector_size * data_vector_entry_size),
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
