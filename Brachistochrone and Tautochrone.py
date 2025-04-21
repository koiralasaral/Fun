import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Cycloid Parameters ---
R = 1.0  # parameter (radius) for the generating circle.
# Define theta parameter values from 0 to 2π.
theta_vals = np.linspace(0, 2*np.pi, 300)
# Standard cycloid equations; invert y to simulate descent.
x_vals = R * (theta_vals - np.sin(theta_vals))
y_vals = -R * (1 - np.cos(theta_vals))

# --- Set Up the Plot ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_vals, y_vals, 'k-', label='Cycloid (Brachistochrone/Tautochrone)')
dot, = ax.plot([], [], 'ro', markersize=8)
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12)
ax.set_xlim(0, np.max(x_vals) + 0.5)
ax.set_ylim(np.min(y_vals) - 0.5, 0.5)
ax.set_aspect('equal')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Cycloid: Brachistochrone & Tautochrone Animation')
plt.legend()

# --- Animation Setup ---
def init():
    dot.set_data([], [])
    time_text.set_text('')
    return dot, time_text

# Assume total time of descent T_total (for demonstration, let T_total = 2 seconds).
T_total = 2.0
def animate(i):
    dot.set_data(x_vals[i], y_vals[i])
    t = i / (len(x_vals) - 1) * T_total
    time_text.set_text(f'Time = {t:.2f} s')
    return dot, time_text

ani = FuncAnimation(fig, animate, frames=len(x_vals), init_func=init,
                    interval=20, blit=True, repeat=True)

plt.show()
