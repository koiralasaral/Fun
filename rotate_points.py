import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Helper function: Rotate points (works on an (N,2) array).
# =============================================================================
def rotate_points(points, theta, pivot=None):
    """
    Rotate an array of 2D points by an angle theta (in radians).
    If pivot is provided, rotate about that point; otherwise, rotate about the origin.

    :param points: NumPy array of shape (N, 2).
    :param theta: Rotation angle in radians.
    :param pivot: Optional point (array-like of shape (2,)).
    :return: Rotated points as a NumPy array.
    """
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    if pivot is not None:
        return (points - pivot) @ R.T + pivot
    else:
        return points @ R.T

# =============================================================================
# Example: Rotating Points about a Pivot and Displaying Rotated Axes
# =============================================================================

# Data: points and pivot (change as desired)
points = np.array([[1, 2], [3, 4], [5, 6]])
pivot = np.array([-2, -3])   # Pivot point for rotation
theta_points = np.pi / 6     # 30° rotation

# --- Print input values ---
print("=== Rotating Points About a Pivot ===")
print("Input Points:")
print(points)
print("Rotation Pivot:", pivot)
print("Rotation Angle (radians):", theta_points)

# Rotate points about the pivot.
rotated_points = rotate_points(points, theta_points, pivot=pivot)

# --- Print sample output ---
print("\nRotated Points:")
print(rotated_points)

# Set up the plot.
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(points[:, 0], points[:, 1], color="blue", label="Original Points")
ax.scatter(rotated_points[:, 0], rotated_points[:, 1],
           color="red", label="Rotated Points (Pivot)")

# Annotate each point.
for i, pt in enumerate(points):
    ax.text(pt[0] + 0.2, pt[1] + 0.2, f"P{i+1}", color="blue", fontsize=10)
for i, pt in enumerate(rotated_points):
    ax.text(pt[0] + 0.2, pt[1] + 0.2, f"P{i+1}'", color="red", fontsize=10)

# Mark the pivot.
ax.scatter(pivot[0], pivot[1], color="green", s=100, marker="x", label="Pivot")

# --- Add rotated axes arrows from the pivot ---
# Compute rotated unit vectors for the new x' and y' axes.
rot_x = np.array([np.cos(theta_points), np.sin(theta_points)])    # Rotated x-axis unit vector.
rot_y = np.array([-np.sin(theta_points), np.cos(theta_points)])   # Rotated y-axis unit vector.

arrow_scale = 4  # Adjust the arrow length as needed.
ax.annotate("", xy=pivot + arrow_scale * rot_x, xytext=pivot,
            arrowprops=dict(facecolor="orange", width=2, headwidth=8))
ax.text(*(pivot + arrow_scale * rot_x + np.array([0.3, 0.2])), "x'", color="orange", fontsize=12)

ax.annotate("", xy=pivot + arrow_scale * rot_y, xytext=pivot,
            arrowprops=dict(facecolor="purple", width=2, headwidth=8))
ax.text(*(pivot + arrow_scale * rot_y + np.array([0.3, 0.2])), "y'", color="purple", fontsize=12)

# Draw global axes as dashed gray lines.
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.axvline(0, color="gray", linestyle="--", linewidth=1)

# Set axis limits so that the arrows are clearly visible.
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 10)

ax.set_title("Points Rotation about a Pivot with Rotated Axes")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.legend(loc="upper left")
ax.grid(True, linestyle="--", linewidth=0.5)
plt.show()
