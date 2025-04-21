import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Recompute key ingredients at θ₀ = π/4.
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
curvature_expr = sp.Abs(dx_dtheta * d2y_dtheta - dy_dtheta * d2x_dtheta) \
                 / (dx_dtheta**2 + dy_dtheta**2)**(sp.Rational(3, 2))
R_curvature_expr = 1 / curvature_expr

# Lambdify for numerical evaluation.
x_func = sp.lambdify(theta, x_expr, 'numpy')
y_func = sp.lambdify(theta, y_expr, 'numpy')
dx_func = sp.lambdify(theta, dx_dtheta, 'numpy')
dy_func = sp.lambdify(theta, dy_dtheta, 'numpy')
R_curvature_func = sp.lambdify(theta, R_curvature_expr, 'numpy')

theta0 = np.pi / 4
P = np.array([x_func(theta0), y_func(theta0)])
dP = np.array([dx_func(theta0), dy_func(theta0)])
T = dP / np.linalg.norm(dP)
N = np.array([-T[1], T[0]])
R0 = R_curvature_func(theta0)
C = P + R0 * N  # Centre of oscillating circle.

# Find intersections of the line through O and C with the circle.
C_norm = np.linalg.norm(C)
if C_norm != 0:
    s1 = 1 + R0 / C_norm
    s2 = 1 - R0 / C_norm
    R1 = s1 * C
    R2 = s2 * C
else:
    R1 = R2 = C

# Generate the osculating circle.
phi = np.linspace(0, 2*np.pi, 200)
circle_x = C[0] + R0 * np.cos(phi)
circle_y = C[1] + R0 * np.sin(phi)

plt.figure(figsize=(8, 8))
plt.plot(circle_x, circle_y, 'g--', label='Osculating Circle')
plt.plot([R1[0], R2[0]], [R1[1], R2[1]], 'r-', linewidth=2, label='Chord through O')
plt.plot(0, 0, 'ko', label='O (Origin)')
plt.plot(C[0], C[1], 'co', markersize=8, label='C (Centre)')
plt.plot(R1[0], R1[1], 'mo', label='R₁')
plt.plot(R2[0], R2[1], 'mo', label='R₂')
plt.axis('equal')
plt.legend()
plt.title('Chord of Curvature Through the Pole (Along OC) with Osculating Circle')
plt.grid(True)
plt.show()
