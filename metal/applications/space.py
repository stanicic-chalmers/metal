from abc import ABC, abstractmethod


class BaseSpace(ABC):
    """BaseSpace is a class for handling a configurational space

    Parameters
    ----------
    dbpath : str
        Path to directory where databases are stored
    """

    def __init__(
        self,
        dbpath
    ):
        self.dbpath = dbpath

    @abstractmethod
    def map(self):
        pass
