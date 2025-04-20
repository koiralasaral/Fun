import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Helper function: Rotate coordinate grids (for mesh grids).
# =============================================================================
def rotate_axes(X, Y, theta):
    """
    Rotate coordinate grid arrays X and Y by angle theta.

    :param X: X coordinates (NumPy array).
    :param Y: Y coordinates (NumPy array).
    :param theta: Rotation angle in radians.
    :return: Tuple (x_rot, y_rot)
    """
    x_rot = X * np.cos(theta) - Y * np.sin(theta)
    y_rot = X * np.sin(theta) + Y * np.cos(theta)
    return x_rot, y_rot

# =============================================================================
# Example: Transforming a Quadratic Equation via Translation (new origin) and Rotation
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# =============================================================================
# Symbolic Calculation of New Equation Coefficients
# =============================================================================
# Define symbols
X, Y = sp.symbols('X Y', real=True)
theta = sp.pi/4

# For a translation taking the pivot (-2,-3) into the new origin,
# the transformation is:
#   x = X*cos(theta) + Y*sin(theta) - 2
#   y = -X*sin(theta) + Y*cos(theta) - 3
cos_theta = sp.cos(theta)
sin_theta = sp.sin(theta)

x_expr = cos_theta*X + sin_theta*Y - 2
y_expr = -sin_theta*X + cos_theta*Y - 3

# Original quadratic equation:
expr = 2*x_expr**2 + 4*x_expr*y_expr - 5*y_expr**2 + 20*x_expr - 22*y_expr - 14

# Expand and collect terms
expr_simpl = sp.simplify(sp.expand(expr))
expr_collected = sp.collect(expr_simpl, [X, Y])

# Convert to a polynomial in X, Y to extract coefficients:
poly_expr = sp.Poly(expr_simpl, X, Y)

coeff_dict = poly_expr.as_dict()
# Coefficients: keys are tuples (i,j) for X^i * Y^j.
A_new = coeff_dict.get((2, 0), 0)    # Coefficient of X^2
B_new = coeff_dict.get((1, 1), 0)    # Coefficient of X*Y
C_new = coeff_dict.get((0, 2), 0)    # Coefficient of Y^2
D_new = coeff_dict.get((1, 0), 0)    # Coefficient of X (linear in X)
E_new = coeff_dict.get((0, 1), 0)    # Coefficient of Y (linear in Y)
F_new = coeff_dict.get((0, 0), 0)    # Constant term

print("=== Quadratic Equation Transformation (Symbolic) ===")
print("Original Equation: 2x^2 + 4xy - 5y^2 + 20x - 22y - 14 = 0")
print("Pivot (new origin): (-2, -3)")
print("Rotation Angle (radians):", theta)
print("\nNew Equation in Rotated Coordinates (X, Y):")
new_eq = sp.pretty(expr_collected)
print(new_eq)
print("\nExtracted Coefficients:")
print("Coefficient of X^2 (A_new):", A_new)      # Expected: 0.5
print("Coefficient of X*Y (B_new):", B_new)        # Expected: -7.0
print("Coefficient of Y^2 (C_new):", C_new)        # Expected: -3.5
print("Coefficient of X (D_new):", D_new)          # Expected: -sqrt(2)
print("Coefficient of Y (E_new):", E_new)          # Expected: -21*sqrt(2)
print("Constant term (F_new):", F_new)

# =============================================================================
# Numerical Transformation: Translation & Rotation of a Mesh Grid
# =============================================================================

# Create a mesh grid in the original (x, y) coordinates.
X_orig, Y_orig = np.meshgrid(np.linspace(-100, 100, 400),
                             np.linspace(-100, 100, 400))
# Original quadratic equation in (x, y):
Z = 2*X_orig**2 + 4*X_orig*Y_orig - 5*Y_orig**2 + 20*X_orig - 22*Y_orig - 14

# Translate the grid so that the pivot becomes the new origin.
# Since pivot = (-2, -3), we have:
#     X_t = x - (-2) = x + 2, and Y_t = y - (-3) = y + 3.
X_t = X_orig + 2
Y_t = Y_orig + 3

# Rotate the translated grid about the new origin:
#  new X = X_t*cos(theta) - Y_t*sin(theta)
#  new Y = X_t*sin(theta) + Y_t*cos(theta)
X_rot, Y_rot = rotate_axes(X_t, Y_t, float(theta))
# (X_rot, Y_rot) are the new (rotated, translated) coordinates.

# To verify, compute the numerical value of the original equation
# by transforming back to the original (x,y) coordinates:
# x_new = X_rot*cos(theta) - Y_rot*sin(theta) - 2  (since inverse translation subtracts 2)
# y_new = X_rot*sin(theta) + Y_rot*cos(theta) - 3  (similarly subtract 3)
x_new = X_rot * np.cos(float(theta)) - Y_rot * np.sin(float(theta)) - 2
y_new = X_rot * np.sin(float(theta)) + Y_rot * np.cos(float(theta)) - 3
Z_rot_new = 2*x_new**2 + 4*x_new*y_new - 5*y_new**2 + 20*x_new - 22*y_new - 14

# --- Print a Sample Grid Point (Original and Transformed) ---
sample_idx = (0, 0)
print("\nSample Original Grid Point:")
print("x =", X_orig[sample_idx], " y =", Y_orig[sample_idx], " Z =", Z[sample_idx])
print("Sample Transformed Grid Point:")
print("X_rot =", X_rot[sample_idx], " Y_rot =", Y_rot[sample_idx],
      " Z_rot_new =", Z_rot_new[sample_idx])

# =============================================================================
# Plotting: Original and Transformed Equation
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the original equation.
ax1.contour(X_orig, Y_orig, Z, levels=[0], colors="blue")
ax1.set_title("Original Equation (x, y)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
ax1.axvline(0, color="gray", linestyle="--", linewidth=1)
ax1.grid(True, linestyle="--", linewidth=0.5)

# Plot the transformed (translated & rotated) equation.
ax2.contour(X_rot, Y_rot, Z_rot_new, levels=[0], colors="red")
ax2.set_title("Transformed Equation in Rotated Coordinates\nwith New Origin (-2, -3)")
ax2.set_xlabel("X (rotated, translated)")
ax2.set_ylabel("Y (rotated, translated)")
ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
ax2.axvline(0, color="gray", linestyle="--", linewidth=1)
ax2.grid(True, linestyle="--", linewidth=0.5)

# --- Add Rotated Axes Arrows on the Transformed Plot (about the new origin) ---
# In the new coordinate system, the origin is at (0,0).
axes_scale = 40  # Adjust scale for visibility.
rot_x_eq_num = np.array([np.cos(float(theta)), np.sin(float(theta))]) * axes_scale
rot_y_eq_num = np.array([-np.sin(float(theta)), np.cos(float(theta))]) * axes_scale

ax2.annotate("", xy=rot_x_eq_num, xytext=(0, 0),
             arrowprops=dict(facecolor="orange", width=3, headwidth=12))
ax2.text(rot_x_eq_num[0] + 2, rot_x_eq_num[1] + 2, "x'", color="orange", fontsize=12)

ax2.annotate("", xy=rot_y_eq_num, xytext=(0, 0),
             arrowprops=dict(facecolor="purple", width=3, headwidth=12))
ax2.text(rot_y_eq_num[0] + 2, rot_y_eq_num[1] + 2, "y'", color="purple", fontsize=12)

ax2.set_xlim(-60, 60)
ax2.set_ylim(-60, 60)

plt.tight_layout()
plt.show()
