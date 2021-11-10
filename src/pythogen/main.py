from typing import Any, Callable
from string import ascii_letters
import numpy as np
from dataclasses import dataclass, field

from crossover import single_point
from mutate import exchange2
from selection import roulette_wheel, tournament

CHAR_VOCAB = np.array(list(ascii_letters))

TARGET = "Hello World"
TARGET_LIST = list(TARGET)
TARGET_INDICES = np.array([TARGET_LIST.index(i) for i in list(TARGET)])


class evaluate_genome:
    def evaluate(self, indices: list[int]) -> float:
        diff = np.sum(np.abs(TARGET_INDICES - indices))
        return diff


def __indices_to_string(indices: list[int]) -> str:
    return "".join(CHAR_VOCAB[indices])


@dataclass
class island:

    population: np.ndarray = field(init=False, repr=False)
    fitness: np.ndarray = field(init=False, repr=False)

    population_size: int
    genome_size: int

    prob_mutation: float = field(default=0.25)
    prob_crossover: float = field(default=0.75)

    random_state: np.random.RandomState = field(default=np.random.RandomState(0))

    def __post_init__(self):
        self.population = self.random_state.randint(
            low=0,
            high=CHAR_VOCAB.shape[0],
            size=(self.population_size, self.genome_size),
        )

        self.fitness = np.apply_along_axis(self.evaluate, 1, self.population)
        # self.fitness = np.zeros((self.population_size, 1), dtype=np.float32)
        # self.fitness = np.random.rand(self.population_size, 1)

    def step(self, init: bool = False):
        parent1 = self.population[self.select()]
        parent2 = self.population[self.select()]

        child1, child2 = self.crossover((parent1, parent2))

        if self.random_state.rand() < self.prob_mutation:
            child1 = self.mutate(child1)
        if self.random_state.rand() < self.prob_mutation:
            child2 = self.mutate(child2)

        fitness1 = self.evaluate(child1)
        fitness2 = self.evaluate(child2)

        print(fitness1)
        print(fitness2)

        # self.population[0] = child1
        # self.population[1] = child2

        return self


def make_island(*args, **kwargs) -> island:
    name = "Island"
    base_class = island
    ops = (roulette_wheel, single_point, exchange2, evaluate_genome)
    cls = type(name, (base_class, *ops), {})

    return cls(**kwargs)


if __name__ == "__main__":
    # configuration = (roulette_wheel, single_point, exchange2, evalute_genome)
    island0 = make_island(population_size=10, genome_size=TARGET_INDICES.shape[0])
    island0.step()