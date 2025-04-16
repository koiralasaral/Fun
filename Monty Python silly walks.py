import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_silly_walk(num_steps=200):
    """
    Generate x and y coordinates for a stochastic random walk.
    The walk takes random steps of a random length at a random angle.
    Every 10 steps, we print a Monty Python–inspired commentary.
    """
    x = [0]
    y = [0]
    
    for i in range(1, num_steps):
        # Choose a random step length (you can adjust the range for more/less intensity)
        step_length = random.uniform(0.5, 2.5)
        # Choose a random angle between 0 and 2π
        angle = random.uniform(0, 2 * math.pi)
        new_x = x[-1] + step_length * math.cos(angle)
        new_y = y[-1] + step_length * math.sin(angle)
        x.append(new_x)
        y.append(new_y)
        
        # Print some Monty Python commentary every 10 steps
        if i % 10 == 0:
            print(f"Step {i}: And now, we execute a most absurd maneuver at {angle:.2f} radians!")
    return x, y

def animate_walk(x, y):
    """
    Animate the generated random walk path using matplotlib.
    The title pays homage to the Ministry of Silly Walks.
    """
    fig, ax = plt.subplots()
    ax.set_title('Monty Python Ministry of Silly Walks')
    # Extend the axes slightly more than the min/max for a better view
    ax.set_xlim(min(x) - 1, max(x) + 1)
    ax.set_ylim(min(y) - 1, max(y) + 1)
    line, = ax.plot([], [], lw=2, marker='o')

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

if __name__ == "__main__":
    print("Welcome to the Monty Python Ministry of Silly Walks Simulator!")
    x, y = generate_silly_walk(num_steps=200)
    animate_walk(x, y)