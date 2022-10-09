from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed


@dataclass
class island:

    population: np.ndarray = field(init=False, repr=False)
    fitness: np.ndarray = field(init=False, repr=False)

    population_size: int
    # genome_size: int

    prob_mutation: float = field(default=0.25)
    prob_crossover: float = field(default=0.75)

    random_state: np.random.RandomState = field(default=np.random.RandomState(0))

    generation: int = field(default=2)

    def __post_init__(self):
        # self.population = self.random_state.randint(
        #     low=0,
        #     high=CHAR_VOCAB.shape[0],
        #     size=(self.population_size, self.genome_size),
        # )
        # pass
        self.create_initial_population()
        self.fitness = np.apply_along_axis(self.evaluate, 1, self.population)

    def step(self, init: bool = False):
        """Performs an iteration; producing a new population."""

        def _step():
            parent1 = self.population[self.select()]
            parent2 = self.population[self.select()]

            child1, child2 = self.crossover((parent1, parent2))

            if self.random_state.rand() < self.prob_mutation:
                child1 = self.mutate(child1)
            if self.random_state.rand() < self.prob_mutation:
                child2 = self.mutate(child2)

            fitness1 = self.evaluate(child1)
            fitness2 = self.evaluate(child2)

            return child1, child2, fitness1, fitness2

        new_population = list()
        new_fitness = list()

        for i in range(0, self.population_size, 2):
            child1, child2, fitness1, fitness2 = _step()
            new_population.append(child1)
            new_population.append(child2)
            new_fitness.append(fitness1)
            new_fitness.append(fitness2)

        self.generation += 1

        return self
