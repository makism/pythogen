from dataclasses import dataclass
from re import A
from string import ascii_letters

import numpy as np

from pythogen import island, genome, single_point, exchange2, roulette_wheel


CHAR_VOCAB = np.array(list(ascii_letters))
TARGET = "Hello World"
TARGET_LIST = list(TARGET)
TARGET_INDICES = np.array([TARGET_LIST.index(i) for i in list(TARGET)])


@dataclass
class ExampleGenome(genome):
    genome_size: int = len(TARGET)

    def create_initial_population(self):
        self.population = self.random_state.randint(
            low=0,
            high=CHAR_VOCAB.shape[0],
            size=(self.population_size, self.genome_size),
        )

    def evaluate(self, indices: list[int]) -> np.float32:
        diff = np.sum(np.abs(TARGET_INDICES - indices), dtype=np.float32)
        return diff


def __indices_to_string(indices: list[int]) -> str:
    return "".join(CHAR_VOCAB[indices])


def make_island(*args, **kwargs) -> island:
    name = "Island"
    base_class = island
    ops = (ExampleGenome, roulette_wheel, single_point, exchange2)
    cls = type(name, (base_class, *ops), {})

    return cls(**kwargs)


if __name__ == "__main__":
    # configuration = (roulette_wheel, single_point, exchange2, evalute_genome)
    island0 = make_island(population_size=10)  # , genome_size=TARGET_INDICES.shape[0])
    island0.step(init=True)

    for i in range(100):
        island0.step()
        print(__indices_to_string(island0.population[0]))
