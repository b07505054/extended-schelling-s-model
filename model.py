
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
import random
import grid_setting, main


def get_moore_neighbors(grid, pos):
    """Get Moore neighborhood (8 neighbors)."""
    i, j = pos
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < main.L and 0 <= nj < main.W and grid[ni, nj] != 0:
                neighbors.append((ni, nj))
    return neighbors

def compute_similarity(attr1, attr2):
    """Compute similarity using Hamming distance."""
    return 1 - hamming(attr1, attr2)

def is_satisfied(grid, agents, agent, tau_u, tau_s):
    """Check if agent is satisfied based on utility and similarity thresholds."""
    neighbors = get_moore_neighbors(grid, agent['pos'])
    if not neighbors:
        return True  # No neighbors, satisfied
    similar_neighbors = 0
    for n_pos in neighbors:
        neighbor_id = grid[n_pos]
        neighbor = next(a for a in agents if a['id'] == neighbor_id)
        similarity = compute_similarity(agent['attributes'], neighbor['attributes'])   
        if  similarity >= tau_s:
            similar_neighbors += 1
    theta = similar_neighbors / len(neighbors)
    return theta >= tau_u

def find_vacant_spot(grid, agent, agents, tau_u, tau_s):
    """Find nearest vacant spot where agent would be satisfied."""
    i, j = agent['pos']
    for r in range(1, max(main.L, main.W)):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if abs(di) != r and abs(dj) != r:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < main.L and 0 <= nj < main.W and grid[ni, nj] == 0:
                    # Temporarily place agent to check satisfaction
                    grid[agent['pos']] = 0
                    grid[ni, nj] = agent['id']
                    agent['pos'] = (ni, nj)
                    satisfied = is_satisfied(grid, agents, agent, tau_u, tau_s)
                    # Revert changes
                    grid[ni, nj] = 0
                    grid[agent['pos']] = agent['id']
                    agent['pos'] = (i, j)
                    if satisfied:
                        return (ni, nj)
    return None

def compute_segregation(grid, agents):
    """Compute segregation level as sum of identical neighbors."""
    segregation = 0
    for agent in agents:
        neighbors = get_moore_neighbors(grid, agent['pos'])
        for n_pos in neighbors:
            neighbor_id = grid[n_pos]
            neighbor = next(a for a in agents if a['id'] == neighbor_id)
            if agent['attributes'][0] == neighbor['attributes'][0]:
                segregation += 1
    return segregation



def simulate(num_attributes, tau_u, tau_s, max_iter, NUM_AGENTS, a1_values, P_A2_GIVEN_A1, COLORS):
    """Run the simulation for the extended Schelling model."""
    grid, agents = grid_setting.initialize_grid(NUM_AGENTS, num_attributes, a1_values, P_A2_GIVEN_A1)
    iteration = 0
    while iteration < max_iter:
        unsatisfied = [a for a in agents if not is_satisfied(grid, agents, a, tau_u, tau_s)]
        if not unsatisfied:
            break
        random.shuffle(unsatisfied)
        for agent in unsatisfied:
            vacant = find_vacant_spot(grid, agent, agents, tau_u, tau_s)
            if vacant:
                grid[agent['pos']] = 0
                grid[vacant] = agent['id']
                agent['pos'] = vacant
        segregation = compute_segregation(grid, agents)
        print(f"segregation in in iteration {iteration}: {segregation}")
        
        if iteration % 100 == 0 or iteration == 0:
            grid_setting.plot_grid(grid, agents, num_attributes, iteration, segregation, COLORS)
        iteration += 1
        segregation = compute_segregation(grid, agents)
        grid_setting.plot_grid(grid, agents, num_attributes, iteration, segregation, COLORS)
    return iteration, segregation

    