import copy
from pathlib import Path
import xlwings as xw


__all__ = (
    "excel"
)


class Excel(object):

    def __init__(self):
        super().__init__()

    def read(
            self,
            file_path: str | Path
    ):
        file_path = Path(file_path)


excel = Excel()


class Sheet(object):

    def __init__(self):
        super().__init__()

    def read(
            self,
            row: int,
            col: int,
    ):
        pass


class Book(object):

    def __init__(
            self,
            file_path: str | Path,
            **kwargs,
    ):
        super().__init__()
        self._kwargs = copy.deepcopy(kwargs)
        file_path = (
            Path(file_path)
            if isinstance(file_path, str)
            else file_path
        )
        self._book = xw.Book(
            fullname=file_path.absolute(),
        )

    def sheet(
            self,
            name: str
    ):
        pass

    def read(self):
        pass

