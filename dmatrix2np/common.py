from enum import Enum
import struct
import sys


class FieldDataType(Enum):
    kFloat32 = 1
    kDouble = 2
    kUInt32 = 3
    kUInt64 = 4


data_type_sizes = {
    'kFloat32': 4,
    'kDouble': 8,
    'kUInt32': 4,
    'kUInt64': 8,
}

size_t_size = struct.calcsize("N")
byte_order_str = sys.byteorder
size_t_dtype_str = f'{"<" if byte_order_str == "little" else ">"}i{size_t_size}'
