

# -*- coding: utf-8 -*-
"""EPSO"""

# Author: Ariana Paola Trujillo Meraz
# Algorithm: Enhanced Particle Swarm Optimization (EPSO) - This algorithm
# is an evolutionary computation technique used for finding approximate solutions
# to optimization problems. It's inspired by the social behavior of bird flocking
# or fish schooling.

#EPSO                                                                                                                                                                                                                                                                                                                                                         1. Exploratory Particles PSO
#At each restart, I measure particle speeds and classify them into: Exploitative: speed below average. Exploratory: speed above average.
#- Exploratory particle detection + restart seeding
#50 best fitness exploratory particles.


import time
import numpy as np
from EvoloPy.solution import solution

# Function to update the local best position based on a ring neighborhood
def update_lbest(costs, positions):
    """Ring-3 neighborhood (left, self, right)."""
    popsize = positions.shape[0]
    lbest = np.zeros_like(positions)
    # Iterate through each particle to determine its local best neighbor
    for i in range(popsize):
        left = (i - 1) % popsize # Get the index of the left neighbor (wraps around)
        right = (i + 1) % popsize # Get the index of the right neighbor (wraps around)
        neighborhood = [left, i, right] # Define the neighborhood indices
        # Find the best particle within the neighborhood based on cost
        best_idx = min(neighborhood, key=lambda idx: costs[idx])
        lbest[i, :] = positions[best_idx, :] # Assign the position of the best neighbor as the local best
    return lbest

# Enhanced PSO (EPSO) optimizer function
def EPSO(objf, lb, ub, dim, PopSize, iters):
    """
    Enhanced PSO (EPSO) optimizer in EvoloPy format.
    """
    # Parameters of the EPSO algorithm
    w = 0.72984 # Inertia weight
    c1 = 2.05 * w # Cognitive coefficient
    c2 = 2.05 * w # Social coefficient

    # Prepare bounds - ensure lb and ub are lists and convert to numpy arrays
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    LB = np.array(lb, dtype=float) # Lower bounds
    UB = np.array(ub, dtype=float) # Upper bounds

    # Define velocity limits based on bounds
    Vmin = -np.abs(UB - LB)
    Vmax = np.abs(UB - LB)

    # Create a solution object to store results in EvoloPy format
    s = solution()
    s.optimizer = "EPSO"
    s.objfname = getattr(objf, "__name__", "objective")
    print(f'EPSO is optimizing "{s.objfname}"')

    # Initialize random number generator
    rng = np.random.default_rng()
    # Initialize particle positions randomly within the bounds
    pos = rng.uniform(LB, UB, size=(PopSize, dim))
    # Initialize particle velocities to zero
    vel = np.zeros((PopSize, dim))

    # Evaluate initial swarm
    pBest = pos.copy() # Personal best position for each particle
    # Evaluate the objective function for each particle
    pBestScore = np.array([objf(pBest[i, :]) for i in range(PopSize)])
    # Find the index of the global best particle
    g_idx = int(np.argmin(pBestScore))
    # Store the global best position
    gBest = pBest[g_idx, :].copy()
    # Store the global best fitness score
    gBestScore = float(pBestScore[g_idx])

    # Initialize local best positions based on the initial swarm
    lbest = update_lbest(pBestScore, pBest)
    # Array to store the best fitness found at each iteration
    convergence_curve = np.zeros(iters, dtype=float)

    # Restart settings for the EPSO algorithm
    # Define iteration points for potential restarts
    restart_points = [int(frac * iters) for frac in [0.2, 0.4, 0.6, 0.8, 1.0]]
    # Define windows for detecting exploratory behavior before each restart
    exploratory_windows = [(r - int(0.0166 * iters), r) for r in restart_points]
    # List to collect promising solutions during exploratory phases
    collected_solutions = []

    # Timing the optimization process
    t0 = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main optimization loop
    for l in range(iters):
        # Velocity & position update
        r1 = rng.random((PopSize, dim)) # Random vector 1 for cognitive component
        r2 = rng.random((PopSize, dim)) # Random vector 2 for social component
        # Update velocity based on inertia, cognitive, and social components
        vel = w * vel + c1 * r1 * (pBest - pos) + c2 * r2 * (lbest - pos)
        # Clip velocities to stay within defined limits
        vel = np.clip(vel, Vmin, Vmax)
        # Update particle positions
        pos = pos + vel

        # Reflecting boundary handling
        # Check for particles exceeding lower bounds
        below = pos < LB
        overshoot = LB - pos
        # Adjust position and velocity for particles below the lower bound
        pos = np.where(below, LB, pos) + below * overshoot
        vel = np.where(below, 0.0, vel)

        # Check for particles exceeding upper bounds
        above = pos > UB
        overshoot = UB - pos
        # Adjust position and velocity for particles above the upper bound
        pos = np.where(above, UB, pos) + above * overshoot
        vel = np.where(above, 0.0, vel)

        # Evaluate the objective function for the new positions
        newCosts = np.array([objf(pos[i, :]) for i in range(PopSize)])

        # pBest / gBest updates
        # Identify particles with improved fitness
        improved = newCosts < pBestScore
        # Update personal best positions and scores for improved particles
        pBest[improved, :] = pos[improved, :]
        pBestScore[improved] = newCosts[improved]

        # Update local best positions based on the updated personal best scores
        lbest = update_lbest(pBestScore, pBest)
        # Find the index of the current global best particle
        g_idx = int(np.argmin(pBestScore))
        # Update the global best position and score if a better solution is found
        if pBestScore[g_idx] < gBestScore:
            gBestScore = float(pBestScore[g_idx])
            gBest = pBest[g_idx, :].copy()

        # Record the global best fitness for this iteration
        convergence_curve[l] = gBestScore

        # Exploratory detection for the restart mechanism
        for start, end in exploratory_windows:
            # Check if the current iteration is within an exploratory window
            if start <= l < end:
                # Calculate the speed of each particle
                speeds = np.linalg.norm(vel, axis=1)
                # Calculate the average speed of the swarm
                avg_speed = np.mean(speeds)
                # Identify particles with speed above the average
                exploratory_indices = np.where(speeds > avg_speed)[0]
                # Get the positions of the exploratory particles
                exploratory_positions = pos[exploratory_indices]
                # Get the fitness of the exploratory particles
                exploratory_fitness = newCosts[exploratory_indices]
                # Sort exploratory particles by fitness
                sorted_indices = np.argsort(exploratory_fitness)
                # Collect the top 50 fittest exploratory particles (or fewer if less than 50)
                top50 = exploratory_positions[sorted_indices[:min(50, len(sorted_indices))]]
                # If at the end of an exploratory window, collect the top solutions
                if l == end - 1:
                    collected_solutions.append(top50)

        # Restart mechanism
        # Check if the current iteration is a restart point
        if (l + 1) in restart_points:
            new_pos = np.zeros_like(pos)
            # Get collected solutions from the exploratory phase
            collected = collected_solutions.pop(0) if collected_solutions else np.empty((0, dim))
            # Calculate the number of remaining particles to re-initialize randomly
            remaining = PopSize - len(collected)
            # Randomly initialize the remaining particles
            random_init = rng.uniform(LB, UB, size=(remaining, dim))
            # Combine collected solutions and randomly initialized particles
            new_pos[:len(collected)] = collected
            new_pos[len(collected):] = random_init

            # Update particle positions, reset velocities, and re-evaluate
            pos = new_pos.copy()
            vel = np.zeros_like(vel)
            pBest = pos.copy()
            pBestScore = np.array([objf(pos[i, :]) for i in range(PopSize)])
            lbest = update_lbest(pBestScore, pos)

        # Print the best fitness at the current iteration
        print(f'At iteration {l + 1} the best fitness is {gBestScore}')

    # Wrap up - finalize the results
    t1 = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = t1 - t0

    # ✅ Final attributes (always well-formed)
    s.best = float(gBestScore) # Store the final global best fitness
    s.bestIndividual = np.array(gBest, dtype=float).tolist() # Store the position of the final global best particle
    s.convergence = np.array(convergence_curve, dtype=float).tolist() # Store the convergence curve

    # Safety: ensure convergence curve has full length by padding if necessary
    if len(s.convergence) < iters:
        s.convergence += [s.best] * (iters - len(s.convergence))

    return s