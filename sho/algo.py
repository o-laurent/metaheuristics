########################################################################
# Algorithms
########################################################################
import numpy as np


def random(func, init, again):
    """Iterative random search template."""
    best_sol = init()
    best_val = func(best_sol)
    val, sol = best_val, best_sol
    i = 0
    while again(i, best_val, best_sol):
        sol = init()
        val = func(sol)
        if val >= best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol


def greedy(func, init, neighb, again):
    """Iterative randomized greedy heuristic template."""
    best_sol = init()
    best_val = func(best_sol)
    val, sol = best_val, best_sol
    i = 1
    while again(i, best_val, best_sol):
        sol = neighb(best_sol)
        val = func(sol)
        # Use >= and not >, so as to avoid random walk on plateus.
        if val >= best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol


def simulated_annealing(func, init, neighb, again, proba, temp, cap: float = 1, stuck: int = 500):
    """
    Simulated annealing heuristic template.

    If the algo is stuck in a local minimum, it restarts.

    The probability of accepting a worse solution can be capped to try to keep first values.
    """
    print("Starting Simulated Annealing")
    best_sol = init()
    best_val = func(best_sol)
    val, sol = best_val, best_sol
    ever_sol, ever_val = best_sol, best_val  # Save the best values ever
    i, update_step, start = 0, 0, 0
    while again(i, best_val, best_sol):
        sol = neighb(best_sol)
        val = func(sol)

        if val >= best_val or np.random.rand() < min(proba(val, best_val, temp(i-start)), cap):
            best_val, best_sol = val, sol
            update_step = i
        elif i - update_step >= stuck:  # stuck -> restart
            start, update_step = i, i
            best_sol = init()
            best_val = func(best_sol)
            val, sol = best_val, best_sol

        if val > ever_val:
            ever_val, ever_sol = val, sol
        i += 1
    print("\tbest val:", ever_val)
    return ever_val, ever_sol  # Return the best values ever


# To finish
def population_based(func, init, again, selection, variation, best, replacement):
    """Population-based stochastic heuristic template"""
    print("Starting population based algorithm")
    best_val = float('inf')
    pop = init()
    i = 0
    while again(i, best_val, best_sol):
        pop_val = func(offsprings)
        parents, parents_val = selection(pop, pop_val)
        offsprings = variation(parents)
        population, population_val = replacement(parents, offsprings)
        best_val = np.min(population_val)
        best_sol = offsprings[np.argmin(offsprings_val)]
        i += 1
    return best_val, best_sol
