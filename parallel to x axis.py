import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Recompute key quantities at θ₀ = π/4.
theta = sp.symbols('theta', real=True)
p_expr = 1 + 0.5 * sp.cos(theta)
dp_expr = sp.diff(p_expr, theta)
r_expr = sp.sqrt(p_expr**2 + dp_expr**2)
x_expr = r_expr * sp.cos(theta)
y_expr = r_expr * sp.sin(theta)
dx_dtheta = sp.diff(x_expr, theta)
dy_dtheta = sp.diff(y_expr, theta)
d2x_dtheta = sp.diff(dx_dtheta, theta)
d2y_dtheta = sp.diff(dy_dtheta, theta)
curvature_expr = sp.Abs(dx_dtheta*sp.diff(y_expr, theta, 2) - dy_dtheta*sp.diff(x_expr, theta, 2)) \
                 / (dx_dtheta**2+dy_dtheta**2)**(sp.Rational(3, 2))
R_curvature_expr = 1 / curvature_expr

x_func = sp.lambdify(theta, x_expr, 'numpy')
y_func = sp.lambdify(theta, y_expr, 'numpy')
dx_func = sp.lambdify(theta, dx_dtheta, 'numpy')
dy_func = sp.lambdify(theta, dy_dtheta, 'numpy')
R_curvature_func = sp.lambdify(theta, R_curvature_expr, 'numpy')

theta0 = np.pi / 4
P = np.array([x_func(theta0), y_func(theta0)])
dP = np.array([dx_func(theta0), dy_func(theta0)])
T = dP / np.linalg.norm(dP)
N_vec = np.array([-T[1], T[0]])
R0 = R_curvature_func(theta0)
C = P + R0 * N_vec  # Centre of osculating circle

# Horizontal chord: intersection of circle with line y = P[1]
disc = R0**2 - (P[1] - C[1])**2
if disc >= 0:
    x_int1 = C[0] + np.sqrt(disc)
    x_int2 = C[0] - np.sqrt(disc)
    M = np.array([x_int1, P[1]])
    N_pt = np.array([x_int2, P[1]])
else:
    M = N_pt = P

# Generate the osculating circle.
phi = np.linspace(0, 2*np.pi, 200)
circle_x = C[0] + R0 * np.cos(phi)
circle_y = C[1] + R0 * np.sin(phi)

plt.figure(figsize=(8, 8))
plt.plot(circle_x, circle_y, 'g--', label='Osculating Circle')
plt.plot([M[0], N_pt[0]], [M[1], N_pt[1]], 'r-', linewidth=2, label='Horizontal Chord (y = P_y)')
plt.plot(P[0], P[1], 'ro', label='P')
plt.plot(C[0], C[1], 'co', label='C (Centre)')
plt.axis('equal')
plt.legend()
plt.title('Chord of Curvature Parallel to x-Axis Through P with Osculating Circle')
plt.grid(True)
plt.show()
