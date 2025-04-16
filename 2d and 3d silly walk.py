import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def generate_2d_silly_walk(num_steps=200):
    """
    Generate x and y coordinates for a 2D stochastic silly walk.
    """
    x, y = [0], [0]
    
    for _ in range(num_steps):
        step_length = random.uniform(0.5, 2.5)  # Random step size
        angle = random.uniform(0, 2 * math.pi)  # Random direction
        x.append(x[-1] + step_length * math.cos(angle))
        y.append(y[-1] + step_length * math.sin(angle))

    return x, y

def generate_3d_silly_walk(num_steps=200):
    """
    Generate x, y, and z coordinates for a 3D stochastic silly walk.
    """
    x, y, z = [0], [0], [0]
    
    for _ in range(num_steps):
        step_length = random.uniform(0.5, 2.5)  # Random step size
        phi = random.uniform(0, 2 * math.pi)    # Azimuthal angle
        theta = random.uniform(0, math.pi)      # Elevation angle
        
        x.append(x[-1] + step_length * math.sin(theta) * math.cos(phi))
        y.append(y[-1] + step_length * math.sin(theta) * math.sin(phi))
        z.append(z[-1] + step_length * math.cos(theta))

    return x, y, z

def plot_silly_walks():
    """
    Plot both 2D and 3D silly walks side by side.
    """
    fig = plt.figure(figsize=(12, 5))

    # 2D Plot
    ax1 = fig.add_subplot(121)
    x2d, y2d = generate_2d_silly_walk()
    ax1.plot(x2d, y2d, marker='o', linestyle='-', alpha=0.7)
    ax1.set_title("Monty Python's Silly Walk (2D)")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")

    # 3D Plot
    ax2 = fig.add_subplot(122, projection='3d')
    x3d, y3d, z3d = generate_3d_silly_walk()
    ax2.plot(x3d, y3d, z3d, marker='o', linestyle='-', alpha=0.7)
    ax2.set_title("Monty Python's Silly Walk (3D)")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")
    ax2.set_zlabel("Z Position")

    plt.show()

# Run the silly walk simulation
plot_silly_walks()