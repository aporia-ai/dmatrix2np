from .dmatrix_stream_parser import DMatrixStreamParser
from .exceptions import InvalidStructure
from .common import FieldDataType, SIZE_T_DTYPE, data_type_sizes, BYTE_ORDER_STR
import numpy as np


class DMatrixStreamParserV1_0_0(DMatrixStreamParser):
    
    kMagic = 0xffffab01
    verstr = 'version:'
    
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
        verstr = self._handle.read(len(self.verstr.encode()))
        if verstr != self.verstr.encode():
            raise InvalidStructure('Invalid verstr')
        self._ver_major = int.from_bytes(self._handle.read(data_type_sizes['int32_t']), BYTE_ORDER_STR)
        self._ver_minor = int.from_bytes(self._handle.read(data_type_sizes['int32_t']), BYTE_ORDER_STR)
        self._ver_patch = int.from_bytes(self._handle.read(data_type_sizes['int32_t']), BYTE_ORDER_STR)
        # TODO: Version validation
    
    def _skip_fields(self):
        num_of_fields = int.from_bytes(self._handle.read(data_type_sizes['uint64_t']), BYTE_ORDER_STR)
        for _ in range(num_of_fields):
            self._skip_field()
    
    def _skip_field(self):
        # Skip field name (pascal string)
        name_size = int.from_bytes(self._handle.read(data_type_sizes['uint64_t']), BYTE_ORDER_STR)
        self._handle.read(name_size)
        
        # Find field type
        field_type = FieldDataType(int.from_bytes(self._handle.read(data_type_sizes['uint8_t']), BYTE_ORDER_STR))
        is_scalar = self._handle.read(data_type_sizes['bool']) == b'\x01'
        
        if is_scalar:
            self._handle.read(data_type_sizes[field_type.name])
        else:
            # Skip shape.first, shape.second
            self._handle.read(2 * data_type_sizes['uint64_t'])
            
            vector_size = int.from_bytes(self._handle.read(data_type_sizes['uint64_t']), BYTE_ORDER_STR)
            
            # Skip vector
            self._handle.read(vector_size * data_type_sizes[field_type.name])
    
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
        add_array = np.repeat(np.arange(0, sizes.size*self.num_col, self.num_col), sizes)
        new_keys = self._data_vector['keys'] + add_array
        matrix[new_keys] = self._data_vector['data']
        self.np_array = matrix.reshape((self.num_row, self.num_col))

    def get_nparray(self):
        return self.np_array
