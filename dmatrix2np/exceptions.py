class DMatrix2NpError(Exception):
    '''Basic exception for errors raised by dmatrix2np'''
    pass


class InvalidStructure(DMatrix2NpError):
    def __init__(self, message='Invalid structure'):
        super().__init__(message)


class UnsupportedVersion(DMatrix2NpError):
    pass


class InvalidInput(DMatrix2NpError):
    def __init__(self, message='Invalid input'):
        super().__init__(message)
