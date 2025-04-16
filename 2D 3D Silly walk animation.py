import random
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def generate_silly_walk_2d(num_steps=100):
    """ Generate a 2D walk with exaggerated and looping steps """
    x, y = [0], [0]
    
    for _ in range(num_steps):
        step_length = random.uniform(1, 5)  # Exaggerated step length
        angle = random.uniform(0, 2 * math.pi)  # Random direction

        # Occasionally loop backward
        if random.random() < 0.15:
            step_length *= -0.5  # Step backward
        
        x.append(x[-1] + step_length * math.cos(angle))
        y.append(y[-1] + step_length * math.sin(angle))

    return x, y

def generate_silly_walk_3d(num_steps=100):
    """ Generate a 3D walk with exaggerated and looping steps """
    x, y, z = [0], [0], [0]
    
    for _ in range(num_steps):
        step_length = random.uniform(1, 5)  # Exaggerated step length
        phi = random.uniform(0, 2 * math.pi)  # Azimuth angle
        theta = random.uniform(0, math.pi)  # Elevation angle

        # Occasionally loop backward
        if random.random() < 0.15:
            step_length *= -0.5  # Step backward

        x.append(x[-1] + step_length * math.sin(theta) * math.cos(phi))
        y.append(y[-1] + step_length * math.sin(theta) * math.sin(phi))
        z.append(z[-1] + step_length * math.cos(theta))

    return x, y, z

def animate_walk_2d(x, y):
    """ Animate the 2D walk """
    fig, ax = plt.subplots()
    ax.set_title("Monty Python's Silly Walk (2D)")
    ax.set_xlim(min(x) - 1, max(x) + 1)
    ax.set_ylim(min(y) - 1, max(y) + 1)
    line, = ax.plot([], [], lw=2, marker='o', color='purple')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        line.set_data(x[:frame], y[:frame])
        return line,

    ani = FuncAnimation(fig, update, frames=len(x), init_func=init, 
                        blit=True, interval=100, repeat=False)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()

def animate_walk_3d(x, y, z):
    """ Animate the 3D walk """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Monty Python's Silly Walk (3D)")
    ax.set_xlim(min(x) - 1, max(x) + 1)
    ax.set_ylim(min(y) - 1, max(y) + 1)
    ax.set_zlim(min(z) - 1, max(z) + 1)
    line, = ax.plot([], [], [], lw=2, marker='o', color='green')

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,

    def update(frame):
        line.set_data(x[:frame], y[:frame])
        line.set_3d_properties(z[:frame])
        return line,

    ani = FuncAnimation(fig, update, frames=len(x), init_func=init, 
                        blit=True, interval=100, repeat=False)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    ax.set_zlabel("Z Position")
    plt.show()

if __name__ == "__main__":
    print("Welcome to the Monty Python Ministry of Silly Walks Simulator!")

    # Generate and animate 2D silly walk
    x2d, y2d = generate_silly_walk_2d()
    animate_walk_2d(x2d, y2d)

    # Generate and animate 3D silly walk
    x3d, y3d, z3d = generate_silly_walk_3d()
    animate_walk_3d(x3d, y3d, z3d)