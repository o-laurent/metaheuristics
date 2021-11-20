import math
import numpy as np
from scipy.stats import qmc

from . import pb

########################################################################
# Objective functions
########################################################################

# Decoupled from objective functions, so as to be used in display.


def to_sensors(sol):
    """Convert a vector of n*2 dimension to an array of n 2-tuples.

    >>> to_sensors([0,1,2,3])
    [(0, 1), (2, 3)]
    """
    assert(len(sol) > 0)
    sensors = []
    for i in range(0, len(sol), 2):
        sensors.append((int(math.floor(sol[i])), int(math.floor(sol[i+1]))))
    return sensors


def cover_sum(sol, domain_width, sensor_range, dim):
    """Compute the coverage quality of the given vector."""
    assert(0 < sensor_range <= math.sqrt(2))
    assert(0 < domain_width)
    assert(dim > 0)
    assert(len(sol) >= dim)
    domain = np.zeros((domain_width, domain_width))
    sensors = to_sensors(sol)
    cov = pb.coverage(domain, sensors, sensor_range*domain_width)
    s = np.sum(cov)
    assert(s >= len(sensors))
    return s


def pop_cover_sum(pop_sol, domain_width, sensor_range, dim):
    """Compute the coverage qualities of the given array."""
    assert(0 < sensor_range <= math.sqrt(2))
    assert(0 < domain_width)
    assert(dim > 0)
    assert(pop_sol.shape[1] >= dim)
    domain = np.zeros((domain_width, domain_width))
    s = np.zeros(pop_sol.shape[0])
    for i, sol in enumerate(pop_sol):
        sensors = to_sensors(sol)
        cov = pb.coverage(domain, sensors, sensor_range*domain_width)
        s[i] = np.sum(cov)
        assert(s[i] >= len(sensors))
    return s


########################################################################
# Initialization
########################################################################

def rand(dim: int, scale: float):
    """Draw a random vector in [0,scale]**dim."""
    return np.random.random(dim) * scale


def rand_init(dim: int, scale: float):
    """" Initialization heuristic. """
    return np.random.random(dim) * 0.7 * scale + 0.15*scale


def rand_init_halt(dim: int, scale: float, center_size: float = 0.7):
    """" Initialization heuristic. """
    sampler = qmc.Halton(d=1, scramble=True)
    sample = sampler.random(n=dim)
    border_size = (1 - center_size)/2
    points = [border_size*scale + center_size*scale*halt[0] for halt in sample]
    return points


def pop_rand(dim: int, scale, pop_size: int):
    """"Draw a random domain containing nb_sensors ones."""
    return np.random.random(pop_size, dim) * scale

########################################################################
# Neighborhood
########################################################################


def neighb_square(sol, scale, domain_width):
    """Draw a random vector in a square of witdh `scale` in [0,1]
    as a fraction of the domain width around the given solution."""
    assert(0 < scale <= 1)
    side = domain_width * scale
    new = sol + (np.random.random(len(sol)) * side - side/2)
    for i, soli in enumerate(sol):
        if soli > 1 or soli < 0:  # Réparation
            sol[i] = (np.random.random()*0.7+0.15) * \
                domain_width  # tirage uniforme
    return new
