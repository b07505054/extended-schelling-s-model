import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
import random
import grid_setting
from model import simulate
L, W = 50, 50 # Grid size

if __name__=="__main__":
#### variable setting 
    # Grid and simulation parameters
      
    POP_DENSITY = 0.8  # population density
    NUM_AGENTS = int(L * W * POP_DENSITY)
    MAX_ITER_2 = 10  # Max iterations for 2-attribute model
    MAX_ITER_3 = 2000  # Max iterations for 3-attribute model
    TAU_U = 0.8  # Utility threshold 
    TAU_S_2 = 0.5  # Similarity threshold for 2-attribute 
    TAU_S_3 = 0.5   # Similarity threshold for 3-attribute 
    
    a1_values = [1,2] # attribure 1 : race 
    P_A2_GIVEN_A1 = {
        1: {'mu': 2, 'sigma': 2, 'min': 0, 'max': 5},  # N(2, 2) for a1=1
        2: {'mu': 1, 'sigma': 2, 'min': 0, 'max': 5}   # N(1, 2) for a1=2
} 
    COLORS = { 1 : 'green', 2: 'red'} # plot colar depends on race only


### simulation
    random.seed(42)  # For reproducibility
    # Run  model
    print("Runningmodel...")
    iter_2, seg_2 = simulate(2, TAU_U, TAU_S_2, MAX_ITER_2, NUM_AGENTS, a1_values, P_A2_GIVEN_A1, COLORS)
    print(f"model converged in {iter_2} iterations with segregation level {seg_2}")
        