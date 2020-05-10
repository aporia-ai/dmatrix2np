class DMatrix2NpError(Exception):
    ''' Basic exception for errors raised by dmatrix2np'''
    pass


class InvalidStructure(DMatrix2NpError):
    def __init__(self, message='Invalid Structure'):
        super().__init__(message)


class UnsupportedVersion(DMatrix2NpError):
    pass
