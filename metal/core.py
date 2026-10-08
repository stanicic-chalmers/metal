from abc import ABC, abstractmethod


class BaseFramework(ABC):
    '''
    BaseFramework is a class for setting up and training a neural network for
    material discovery

    Args:
        db_path (str): Path to directory where databases are stored

    Raises:
        UserWarning:
        FileNotFoundError:
    '''

    def __init__(
        self,
        db_path=None
    ):
        self.db_path = db_path

    @abstractmethod
    def _train(self):
        pass
