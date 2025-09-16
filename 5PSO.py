
# -*- coding: utf-8 -*-
"""5PSO"""

# Author: Ariana Paola Trujillo Meraz
# Algorithm: 5 Restarts Particle Swarm Optimization (5PSO)
#5PSO                                                                                                                                                                                                                                                                                                                                                         1. Exploratory Particles PSO
#In this variant, PSO is executed for 1200 iterations, followed by a full restart of the swarm.
#This process repeats five times, for a total of 6000 iterations.
#5PSO tests whether performance improves with periodic restarts, even when swarm knowledge is discarded.


import random
import numpy
from EvoloPy.solution import solution
import time


#In this case the function must be all letters, FPSO is the same as 5PSO.

def FPSO(objf, lb, ub, dim, PopSize, iters):
    """
    Particle Swarm Optimization (PSO) with periodic restarts (5PSO).

    Args:
        objf: The objective function to optimize.
        lb: The lower bound of the search space.
        ub: The upper bound of the search space.
        dim: The dimension of the search space.
        PopSize: The number of particles in the swarm.
        iters: The number of iterations.

    Returns:
        A solution object containing the results of the optimization.
    """

    # PSO Parameters
    Vmax = 6  # Maximum velocity
    wMax = 0.9  # Maximum inertia weight
    wMin = 0.2  # Minimum inertia weight
    c1 = 2  # Cognitive parameter
    c2 = 2  # Social parameter

    s = solution()
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    ######################## Initialization
    def init_population():
        """Initializes the particle swarm."""
        vel = numpy.zeros((PopSize, dim))
        pBestScore = numpy.full(PopSize, float("inf"))
        pBest = numpy.zeros((PopSize, dim))
        gBest = numpy.zeros(dim)
        gBestScore = float("inf")
        pos = numpy.zeros((PopSize, dim))
        for i in range(dim):
            pos[:, i] = numpy.random.uniform(0, 1, PopSize) * (ub[i] - lb[i]) + lb[i]
        return pos, vel, pBest, pBestScore, gBest, gBestScore

    pos, vel, pBest, pBestScore, gBest, gBestScore = init_population()
    convergence_curve = numpy.zeros(iters)

    print('PSO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Restart points for periodic restarts
    restart_points = [int(frac * iters) for frac in [0.2, 0.4, 0.6, 0.8, 1.0]]

    for l in range(iters):
        # ---------- Evaluation ----------
        for i in range(PopSize):
            for j in range(dim):
                pos[i, j] = numpy.clip(pos[i, j], lb[j], ub[j])

            fitness = objf(pos[i, :])

            if pBestScore[i] > fitness:
                pBestScore[i] = fitness
                pBest[i, :] = pos[i, :].copy()

            if gBestScore > fitness:
                gBestScore = fitness
                gBest = pos[i, :].copy()

        # ---------- Inertia Weight ----------
        w = wMax - l * ((wMax - wMin) / iters)

        # ---------- Update Velocity and Position ----------
        for i in range(PopSize):
            for j in range(dim):
                r1 = random.random() # Random number for cognitive component
                r2 = random.random() # Random number for social component
                vel[i, j] = (
                    w * vel[i, j]
                    + c1 * r1 * (pBest[i, j] - pos[i, j])
                    + c2 * r2 * (gBest[j] - pos[i, j])
                )

                # Apply velocity clamping
                if vel[i, j] > Vmax:
                    vel[i, j] = Vmax
                if vel[i, j] < -Vmax:
                    vel[i, j] = -Vmax

                pos[i, j] = pos[i, j] + vel[i, j]

        convergence_curve[l] = gBestScore

        print(f"At iteration {l+1} the best fitness is {gBestScore}")

        # ---------- Restart ----------
        if (l + 1) in restart_points:
            # Implement restart logic here. For now, re-initialize the population.
            pos, vel, pBest, pBestScore, gBest, gBestScore = init_population()

    # ================= Wrap up =================
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "FPSO"
    s.bestIndividual = gBest
    s.objfname = objf.__name__

    return s