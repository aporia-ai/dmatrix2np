from .dmatrix_stream_parser import DMatrixStreamParser
from .exceptions import InvalidStructure
from .common import (FieldDataType, data_type_sizes, FLAG_STRUCT, VERSION_STRUCT, VECTOR_SIZE_STRUCT,
                    FIELD_TYPE_STRUCT, KMAGIC_STRUCT)
from os import SEEK_CUR


class DMatrixStreamParserV1_0_0(DMatrixStreamParser):
    
    kMagic = 0xffffab01
    verstr = 'version:'
    
    def __init__(self, buffer_reader, num_row, num_col):
        self._handle = buffer_reader
        self.num_row = num_row
        self.num_col = num_col

    def parse(self):
        self._handle.seek(0)
        self._parse_magic()
        self._parse_version()
        self._skip_fields()
        self._parse_offset_vector()
        self._parse_data_vector()
        return self._get_nparray()
    
    def _parse_magic(self):
        kMagic, = self._read_struct(KMAGIC_STRUCT)
        if kMagic != self.kMagic:
            raise InvalidStructure('Invalid magic')
    
    def _parse_version(self):
        verstr = self._handle.read(len(self.verstr.encode()))
        if verstr != self.verstr.encode():
            raise InvalidStructure('Invalid verstr')
        self._version = self._read_struct(VERSION_STRUCT)

    def _skip_fields(self):
        fields_count, = self._read_struct(VECTOR_SIZE_STRUCT)
        for _ in range(fields_count):
            self._skip_field()
    
    def _skip_field(self):
        # Skip field name (pascal string)
        name_size, = self._read_struct(VECTOR_SIZE_STRUCT)
        self._handle.seek(name_size, SEEK_CUR)
        
        # Find field type
        field_type = FieldDataType(self._read_struct(FIELD_TYPE_STRUCT)[0])
        is_scalar, = self._read_struct(FLAG_STRUCT)
        
        if is_scalar:
            self._handle.seek(data_type_sizes[field_type.name], SEEK_CUR)
        else:
            # Skip shape.first, shape.second
            self._handle.seek(2 * data_type_sizes['uint64_t'], SEEK_CUR)

            vector_size, = self._read_struct(VECTOR_SIZE_STRUCT)
            
            # Skip vector
            self._handle.seek(vector_size * data_type_sizes[field_type.name], SEEK_CUR)
