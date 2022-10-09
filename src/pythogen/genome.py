from abc import abstractmethod

import numpy as np


class genome:
    @abstractmethod
    def create_initial_population(self):
        ...

    @abstractmethod
    def generate(self):
        ...

    @abstractmethod
    def evaluate(self, indices: list[int]) -> np.float32:
        ...
