import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define the symbolic variable and the pedal function.
theta = sp.symbols('theta', real=True)
p_expr = 1 + 0.5 * sp.cos(theta)
dp_expr = sp.diff(p_expr, theta)
d2p_expr = sp.diff(p_expr, theta, 2)

# Radius of curvature in tangential polar form:
R_expr = (p_expr**2 + dp_expr**2)**(sp.Rational(3, 2)) / sp.Abs(p_expr**2 + 2*dp_expr**2 - p_expr*d2p_expr)
R_func = sp.lambdify(theta, R_expr, 'numpy')

# (A) Plot R(θ) versus θ.
theta_vals = np.linspace(0, 2*np.pi, 400)
R_vals = R_func(theta_vals)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(theta_vals, R_vals, label='R(θ)')
plt.xlabel('θ (radians)')
plt.ylabel('Radius of Curvature R(θ)')
plt.title('Radius of Curvature vs. θ')
plt.legend()
plt.grid(True)

# (B) For a sample angle, draw the corresponding curve segment and its osculating circle.
# The Cartesian form of the pedal curve is obtained via:
#   r = sqrt(p(θ)^2 + [p'(θ)]^2) and then x = r*cosθ, y = r*sinθ.
r_expr = sp.sqrt(p_expr**2 + dp_expr**2)
x_expr = r_expr * sp.cos(theta)
y_expr = r_expr * sp.sin(theta)
x_func = sp.lambdify(theta, x_expr, 'numpy')
y_func = sp.lambdify(theta, y_expr, 'numpy')

# Choose a construction point:
theta0 = np.pi / 4
P = np.array([x_func(theta0), y_func(theta0)])

# Compute tangent vector at P (via dP/dθ)
dx_dtheta = sp.diff(x_expr, theta)
dy_dtheta = sp.diff(y_expr, theta)
dx_func = sp.lambdify(theta, dx_dtheta, 'numpy')
dy_func = sp.lambdify(theta, dy_dtheta, 'numpy')
dP = np.array([dx_func(theta0), dy_func(theta0)])
T = dP / np.linalg.norm(dP)  # Unit tangent vector
N = np.array([-T[1], T[0]])   # Unit normal vector

# For consistency, compute the curvature (via the Cartesian formula) here:
d2x_dtheta = sp.diff(dx_dtheta, theta)
d2y_dtheta = sp.diff(dy_dtheta, theta)
curvature_expr = sp.Abs(dx_dtheta * d2y_dtheta - dy_dtheta * d2x_dtheta) / (dx_dtheta**2 + dy_dtheta**2)**(sp.Rational(3, 2))
curvature_func = sp.lambdify(theta, curvature_expr, 'numpy')
k0 = curvature_func(theta0)
R0 = 1/k0 if k0 != 0 else np.inf  # Radius of curvature at θ₀

# Osculating circle: center C = P + R0 * N (here we take the normal as computed).
C = P + R0 * N
phi = np.linspace(0, 2*np.pi, 200)
circle_x = C[0] + R0 * np.cos(phi)
circle_y = C[1] + R0 * np.sin(phi)

# Also plot a short segment of the curve near P.
theta_near = np.linspace(theta0 - 0.5, theta0 + 0.5, 200)
plt.subplot(1, 2, 2)
plt.plot(x_func(theta_near), y_func(theta_near), 'b-', label='Curve')
plt.plot(P[0], P[1], 'ro', label='P')
plt.plot(circle_x, circle_y, 'g--', label='Osculating Circle')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Osculating Circle at P (θ = π/4)')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
