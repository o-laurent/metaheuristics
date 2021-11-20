import math
import numpy as np
import copy
from scipy.stats import qmc

from . import x, y, pb

########################################################################
# Objective functions
########################################################################


def cover_sum(sol, domain_width, sensor_range, dim):
    """Compute the coverage quality of the given array of bits."""
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
    """Compute the coverage quality of the given array of bits."""
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


def to_sensors(sol):
    """Convert an square array of d lines/columns containing n ones
    to an array of n 2-tuples with related coordinates.

    >>> to_sensors([[1,0],[1,0]])
    [(0, 0), (0, 1)]
    """
    assert(len(sol) > 0)
    ones = np.where(sol == 1)
    points = [(ones[0][i], ones[1][i]) for i in range(len(ones[0]))]
    return points


########################################################################
# Initialization
########################################################################

def rand(domain_width, nb_sensors):
    """" Draw a random domain containing nb_sensors ones. """
    domain = np.zeros((domain_width, domain_width))
    for x, y in np.random.randint(0, domain_width, (nb_sensors, 2)):
        domain[y][x] = 1
    return domain


def rand_init(domain_width, nb_sensors):
    """" Initialization heuristic. """
    domain = np.zeros((domain_width, domain_width))
    for x, y in np.random.randint(int(0.15*domain_width), int(0.85*domain_width), (nb_sensors, 2)):
        domain[y][x] = 1
    return domain


def rand_init_halt(domain_width, nb_sensors, center_size: float = 0.7):
    """" Initialization heuristic using Halton sequences (high discrepancy random numbers). """
    domain = np.zeros((domain_width, domain_width))
    sampler = qmc.Halton(d=2, scramble=True)
    sample = sampler.random(n=nb_sensors)
    border_size = (1 - center_size)/2
    points = [(int(border_size*domain_width + center_size*domain_width*halt[0]),
               int(border_size*domain_width + center_size*domain_width*halt[1])) for halt in sample]
    for x, y in points:
        domain[y][x] = 1
    return domain


def pop_rand(domain_width, nb_sensors, pop_size):
    """"Draw a random domain containing nb_sensors ones."""
    domain = np.zeros((pop_size, domain_width, domain_width))
    for ind in range(pop_size):
        for x, y in np.random.randint(0, domain_width, (nb_sensors, 2)):
            domain[ind][y][x] = 1
    return domain

########################################################################
# Neighborhood
########################################################################


def neighb_square(sol, scale, domain_width):
    """Draw a random array by moving every ones to adjacent cells."""
    assert(0 < scale <= 1)
    # Copy, because Python pass by reference
    # and we may not want to alter the original solution.
    new = copy.copy(sol)
    ones = np.where(sol == 1)
    points = [(ones[0][i], ones[1][i]) for i in range(len(ones[0]))]
    for py,px in points:
        # Indices order is (y,x) in order to match
        # coordinates of images (row,col).
        # Add a one somewhere around.
        w = scale/2 * domain_width
        ny = np.random.randint(py-w, py+w)
        nx = np.random.randint(px-w, px+w)
        ny = min(max(0, ny), domain_width-1)
        nx = min(max(0, nx), domain_width-1)

        if new[ny][nx] != 1:
            new[py][px] = 0  # Remove original position.
            new[ny][nx] = 1
        # else pass
    return new
