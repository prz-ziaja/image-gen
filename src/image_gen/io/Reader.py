from abc import ABC, abstractmethod


class ReaderAbstract(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def read_metadata(self, path):
        pass

    @abstractmethod
    def read_image(self, path):
        pass
