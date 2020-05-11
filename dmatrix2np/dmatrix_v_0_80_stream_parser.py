from .dmatrix_stream_parser import DMatrixStreamParser
from .exceptions import InvalidStructure
from .common import data_type_sizes, SIMPLE_VERSION_STRUCT, VECTOR_SIZE_STRUCT, KMAGIC_STRUCT
from os import SEEK_CUR


class DMatrixStreamParserV0_80(DMatrixStreamParser):

    NUM_OF_SCALAR_FIELDS = 3
    kMagic = 0xffffab01
    kVersion = 2

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
        version, = self._read_struct(SIMPLE_VERSION_STRUCT)
        if version != self.kVersion:
            raise InvalidStructure('Invalid version')

    def _skip_fields(self):
        # Skip num_row_, num_col_, num_nonzero_ (all uint64_t)
        self._handle.seek(self.NUM_OF_SCALAR_FIELDS * data_type_sizes['uint64_t'], SEEK_CUR)

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
            vector_size, = self._read_struct(VECTOR_SIZE_STRUCT)
            self._handle.read(vector_size * vector_entry_size)
