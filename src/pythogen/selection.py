from typing import Any, List
from dataclasses import dataclass, field
import numpy as np


@dataclass
class roulette_wheel:
    """Roulette Wheel"""

    def select(self) -> int:
        """Roulette wheel."""
        cutoff = np.sum(self.fitness) * np.random.uniform(0, 1)
        for index, fitness in enumerate(self.fitness):
            cutoff -= fitness
            if cutoff <= 0:
                return index


@dataclass
class tournament:
    """Tournament

    Select the best individual from a population.
    """

    num_participants: int = field(default=10)

    def select(self) -> int:
        random_indices = self.random_state.choice(
            self.population_size, self.num_participants, replace=False
        )

        return np.argmax(self.fitness[random_indices], axis=0)


@dataclass
class elitism:
    """Elitism

    Select the best `k` individuals from a population.
    """

    num_individuals: int = field(default=5)

    def select(self) -> List[int]:
        return np.argsort(self.fitness)[-self.num_individuals :]
