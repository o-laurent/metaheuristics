import numpy as np


def crossover_mutations(population, population_size: int, bit):
    if bit:
        raise NotImplementedError()
    final_pop = np.zeros(population)
    for i in range(population_size):
        ind1 = np.random.randint(0, population_size)
        ind2 = np.random.randint(0, population_size)
        final_pop[i] = np.mean(ind1, ind2)
