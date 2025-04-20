import numpy as np
import matplotlib.pyplot as plt

def transform_points(points, drone_pose):
    """
    Transform points from the drone's local coordinate frame to the global coordinate frame.
    
    :param points: numpy array of shape (N, 2) containing the sensor data in the local frame.
    :param drone_pose: numpy array [x, y, yaw] representing the drone's pose in the world.
                       (x,y) is the drone's position and yaw is its heading (in radians).
    :return: numpy array of shape (N, 2) with points transformed to the world coordinate system.
    """
    x, y, yaw = drone_pose
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw),  np.cos(yaw)]])
    return points @ R.T + np.array([x, y])

# --------------------------------------------------------------
# Example: Drone Sensor Data Transformation
# --------------------------------------------------------------

# Define the drone's pose in world coordinates.
# For example, the drone is at (10, 15) with a heading of 60° (π/3 radians).
drone_pose = np.array([10, 15, np.pi/3])

# Define some sensor measurements from the drone’s local coordinate frame.
# These could represent, for example, distances (and directions) to landmarks.
local_sensor_points = np.array([
    [3, 0],     # 3 meters ahead
    [4, -1],    # further ahead and slightly to the right
    [2, 2],     # a landmark ahead and to the left
    [0, 3]      # directly to the left (in the drone's local frame, y is to the left)
])

# Transform the sensor points to the global coordinate frame.
global_sensor_points = transform_points(local_sensor_points, drone_pose)

# --- Print Input and Output ---
print("Drone Pose (x, y, yaw):", drone_pose)
print("Local Sensor Points:\n", local_sensor_points)
print("Global Sensor Points after Transformation:\n", global_sensor_points)

# --------------------------------------------------------------
# Plotting: Drone, its local coordinate axes, and sensor data in global coordinates
# --------------------------------------------------------------

plt.figure(figsize=(8, 8))

# Plot transformed sensor data in the global coordinate frame.
plt.scatter(global_sensor_points[:, 0], global_sensor_points[:, 1],
            color='red', s=80, label='Sensor Points (Global)')

# Plot the drone's position.
plt.scatter(drone_pose[0], drone_pose[1],
            color='blue', marker='^', s=150, label='Drone Position')

# Draw an arrow representing the drone's heading.
heading_length = 4  # arrow length
heading_vector = np.array([heading_length * np.cos(drone_pose[2]),
                           heading_length * np.sin(drone_pose[2])])
plt.arrow(drone_pose[0], drone_pose[1],
          heading_vector[0], heading_vector[1],
          color='blue', width=0.2, head_width=0.8, length_includes_head=True,
          label='Drone Heading')

# Draw the drone's local coordinate axes.
# In the drone's local frame, define the x-axis (forward) and y-axis (to the left).
local_x_axis = np.array([np.cos(drone_pose[2]), np.sin(drone_pose[2])])
local_y_axis = np.array([-np.sin(drone_pose[2]), np.cos(drone_pose[2])])
axis_length = 3
plt.arrow(drone_pose[0], drone_pose[1],
          axis_length * local_x_axis[0], axis_length * local_x_axis[1],
          color='green', width=0.15, head_width=0.5, length_includes_head=True,
          label='Drone X-axis')
plt.arrow(drone_pose[0], drone_pose[1],
          axis_length * local_y_axis[0], axis_length * local_y_axis[1],
          color='orange', width=0.15, head_width=0.5, length_includes_head=True,
          label='Drone Y-axis')

# Annotate each transformed sensor point with its local coordinate.
for local_pt, global_pt in zip(local_sensor_points, global_sensor_points):
    plt.text(global_pt[0]+0.5, global_pt[1]+0.5,
             f"{local_pt}", color='black', fontsize=10)

# Draw global axes.
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)

plt.xlabel("X (Global Coordinates)")
plt.ylabel("Y (Global Coordinates)")
plt.title("Drone Sensor Data Transformation from Local to Global Coordinates")
plt.legend(loc="upper left")
plt.grid(True)
plt.axis("equal")
plt.show()
