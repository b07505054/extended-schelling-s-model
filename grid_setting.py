import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
import random, main
from scipy.stats import truncnorm

def initialize_grid(num_agents, num_attributes, a1_values, P_A2_GIVEN_A1):
    """Initialize grid with agents and attributes."""
    grid = np.zeros((main.L, main.W), dtype=int)  # 0 for empty
    agent_positions = random.sample([(i, j) for i in range(main.L) for j in range(main.W)], num_agents)
    agents = []
    agent_id = 1
    
    # Generate possible attribute combinations
    if num_attributes == 2:
        
        agents_per_a1 = num_agents // len(a1_values)
        remaining_agents = num_agents % len(a1_values)
        
        for a1 in a1_values:
            num_for_a1 = agents_per_a1 + (1 if remaining_agents > 0 else 0)
            remaining_agents -= 1 if remaining_agents > 0 else 0

            # Generate a2 based on conditional probability P(a2|a1)
            mu, sigma = P_A2_GIVEN_A1[a1]['mu'], P_A2_GIVEN_A1[a1]['sigma']
            a, b = (0 - mu) / sigma, (5 - mu) / sigma  # Standardize bounds
            a2_values = np.round(truncnorm.rvs(a, b, loc=mu, scale=sigma, size=num_for_a1), 0).astype(int)
            a2_values = np.clip(a2_values, 0, 5)
            # print(a2_values)
            for a2 in a2_values:
                if agent_positions:
                    pos = agent_positions.pop()
                    attr = (a1, a2)
                    grid[pos] = agent_id
                    agents.append({'id': agent_id, 'pos': pos, 'attributes': attr})
                    agent_id += 1
        
    return grid, agents

def plot_grid(grid, agents, num_attributes, iteration, segregation,COLORS):
    """Plot the grid with color-coded agents."""
    
    # Colors for agent types  
    
    plt.figure(figsize=(10, 8))
    image = np.zeros((main.L, main.W, 3))
    for i in range(main.L):
        for j in range(main.W):
            if grid[i, j] == 0:
                image[i, j] = [1, 1, 1]  # White for empty
            else:
                agent = next(a for a in agents if a['id'] == grid[i, j])
                attr = agent['attributes'][0]
                color = COLORS[attr]
                if color == 'green':
                    image[i, j] = [0, 1, 0]
                elif color == 'yellow':
                    image[i, j] = [1, 1, 0]
                elif color == 'red':
                    image[i, j] = [1, 0, 0]
                elif color == 'blue':
                    image[i, j] = [0, 0, 1]
                elif color == 'orange':
                    image[i, j] = [1, 0.5, 0]
                elif color == 'purple':
                    image[i, j] = [0.5, 0, 0.5]
                elif color == 'grey':
                    image[i, j] = [0.5, 0.5, 0.5]
                elif color == 'cyan':
                    image[i, j] = [0, 1, 1]
    plt.imshow(image)
    plt.title(f"Iteration {iteration}, Segregation Level: {segregation}")
    plt.axis('off')
    plt.savefig(f'schelling_iter_{iteration}.png')
    plt.close()