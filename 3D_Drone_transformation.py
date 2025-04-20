import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # enables 3D plotting
from matplotlib.animation import FuncAnimation

# --------------------------------------------------------------
# Simulation Parameters
# --------------------------------------------------------------

# Drone's initial pose in world coordinates:
# Format: [x, y, z, yaw, pitch, roll]
# For simplicity we assume constant orientation.
drone_pose = np.array([0.0, 0.0, 10.0, np.pi/6, 0.0, 0.0])
print("Initial Drone Pose (global):", drone_pose)

# Time vector
num_frames = 200
t = np.linspace(0, 20, num_frames)

# Define a moving target path in global coordinates.
# The target follows an oscillatory path in x-y and varying height in z.
target_positions = np.column_stack((
    5 * np.sin(0.2 * t),      # x-coordinate
    5 * np.cos(0.2 * t),      # y-coordinate
    2 * np.sin(0.1 * t) + 5   # z-coordinate (elevated for visibility)
))
print("First Target Position (global):", target_positions[0])
print("Last Target Position (global):", target_positions[-1])

# Proportional gain for the drone to move toward the target.
alpha = 0.05

# --------------------------------------------------------------
# Set Up the 3D Plot
# --------------------------------------------------------------

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(0, 20)
ax.set_xlabel("X (Global)")
ax.set_ylabel("Y (Global)")
ax.set_zlabel("Z (Global)")
ax.set_title("Drone Tracking a Moving Target in 3D")

# Initialize plot elements (with empty data)
drone_plot, = ax.plot([], [], [], 'bo', markersize=8, label="Drone")
target_plot, = ax.plot([], [], [], 'ro', markersize=8, label="Target")
path_plot, = ax.plot([], [], [], 'g--', linewidth=2, label="Target Path")

# --------------------------------------------------------------
# Update Function for Animation
# --------------------------------------------------------------
def update(frame):
    global drone_pose

    # Get the current target position from our simulated path.
    current_target = target_positions[frame]
    
    # Update the drone's position using a proportional step toward the target.
    current_pos = drone_pose[:3]
    new_pos = current_pos + alpha * (current_target - current_pos)
    drone_pose[:3] = new_pos

    # Print sample output every 50 frames.
    if frame % 50 == 0:
        print(f"Frame {frame:3d}:")
        print("   Drone Position (x,y,z):", drone_pose[:3])
        print("   Current Target Position:", current_target)
    
    # Update plot data.
    # For the markers, we pass the positions as single-element lists.
    drone_plot.set_data([drone_pose[0]], [drone_pose[1]])
    drone_plot.set_3d_properties([drone_pose[2]])
    
    target_plot.set_data([current_target[0]], [current_target[1]])
    target_plot.set_3d_properties([current_target[2]])
    
    # Update the line representing the target path (up to current frame)
    path_plot.set_data(target_positions[:frame, 0], target_positions[:frame, 1])
    path_plot.set_3d_properties(target_positions[:frame, 2])
    
    return drone_plot, target_plot, path_plot

# --------------------------------------------------------------
# Run the Animation
# --------------------------------------------------------------
ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
ax.legend()
plt.show()
