import numpy as np


class single_point:
    def crossover(self, parents: list):
        parent1, parent2 = parents
        cutoff = self.random_state.randint(low=0, high=self.genome_size)

        child1 = np.hstack((parent1[0:cutoff], parent2[cutoff:]))
        child2 = np.hstack((parent2[0:cutoff], parent1[cutoff:]))

        return child1, child2


class two_point:
    pass


class uniform:
    pass


class shuffle:
    pass


class three_parent:
    pass