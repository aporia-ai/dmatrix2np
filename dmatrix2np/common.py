from enum import Enum
import struct
import sys


class FieldDataType(Enum):
    """
    This Enum provides an integer translation for the data type corresponding to the 'DataType' enum
    on '/include/xgboost/data.h' file in the XGBoost project
    """
    kFloat32 = 1
    kDouble = 2
    kUInt32 = 3
    kUInt64 = 4
    kStr = 5


# Dictionary of data types size in bytes
data_type_sizes = {
    'kFloat32': 4,
    'kDouble': struct.calcsize("d"),
    'kUInt32': 4,
    'kUInt64': 8,
    'kStr': struct.calcsize('s'),
    'bool': 1,
    'uint8_t': 1,
    'int32_t': 4,
    'uint32_t': 4,
    'uint64_t': 8,
    'int': struct.calcsize("i"),
    'float': struct.calcsize("f"),
    'double': struct.calcsize("d"),
    'size_t': struct.calcsize("N"),
}

SIZE_T_DTYPE = f'{"<" if sys.byteorder == "little" else ">"}i{data_type_sizes["size_t"]}'
VERSION_STRUCT = struct.Struct('iii')
SIMPLE_VERSION_STRUCT = struct.Struct('I')
VECTOR_SIZE_STRUCT = struct.Struct('=Q')
FIELD_TYPE_STRUCT = struct.Struct('b')
FLAG_STRUCT = struct.Struct('?')
KMAGIC_STRUCT = struct.Struct('I')
