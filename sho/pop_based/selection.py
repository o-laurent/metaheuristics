import numpy as np

#aléatoire 
#tournois etc

def pick(population, population_val, pop_size, percentage): 
    offsprings_nb = int(pop_size*percentage/100)
    return population[population_val.argsort()[:offsprings_nb]]