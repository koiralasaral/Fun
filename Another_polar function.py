import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define the tangential polar function and derived expressions
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

# Lambdify all necessary functions
x_func = sp.lambdify(theta, x_expr, 'numpy')
y_func = sp.lambdify(theta, y_expr, 'numpy')
dx_func = sp.lambdify(theta, dx_dtheta, 'numpy')
dy_func = sp.lambdify(theta, dy_dtheta, 'numpy')
R_curvature_func = sp.lambdify(theta, R_curvature_expr, 'numpy')

# Compute the full curve over [0, 2π]
theta_vals = np.linspace(0, 2 * np.pi, 400)
curve_x = x_func(theta_vals)
curve_y = y_func(theta_vals)

plt.figure(figsize=(10, 10))
plt.plot(curve_x, curve_y, 'b-', label='Curve')

# Select a construction point: θ₀ = π/4
theta0 = np.pi / 4
P = np.array([x_func(theta0), y_func(theta0)])
dP = np.array([dx_func(theta0), dy_func(theta0)])
T = dP / np.linalg.norm(dP)
N_vec = np.array([-T[1], T[0]])
R_val = R_curvature_func(theta0)
C = P + R_val * N_vec

# Foot Q (projection of O onto the tangent line at P): Q = (P·T)* T
Q = (np.dot(P, T)) * T

# Tangent line at P: P + s·T
s_vals = np.linspace(-1, 1, 100)
tangent_line = np.array([P + s * T for s in s_vals])

# Osculating circle (centre C, radius R_val)
phi = np.linspace(0, 2 * np.pi, 200)
circle_x = C[0] + R_val * np.cos(phi)
circle_y = C[1] + R_val * np.sin(phi)

# Chord of curvature through O: intersections of line O–C with the circle.
C_norm = np.linalg.norm(C)
if C_norm != 0:
    s1 = 1 + R_val / C_norm
    s2 = 1 - R_val / C_norm
    R1 = s1 * C
    R2 = s2 * C
else:
    R1 = R2 = C

# Horizontal chord through P: solve (x – Cx)^2 = R_val^2 - (P_y – C_y)^2 with y = P_y.
disc = R_val**2 - (P[1] - C[1])**2
if disc >= 0:
    x_int1 = C[0] + np.sqrt(disc)
    x_int2 = C[0] - np.sqrt(disc)
    M = np.array([x_int1, P[1]])
    N_pt = np.array([x_int2, P[1]])
else:
    M = N_pt = P

# Define OX and OY as the x-/y-projections of Q
OX = np.array([Q[0], 0])
OY = np.array([0, Q[1]])

# Plotting all elements:
plt.plot(P[0], P[1], 'ro', markersize=8, label='P')
plt.plot(Q[0], Q[1], 'ko', markersize=8, label='Q (OQ = p)')
plt.plot([0, Q[0]], [0, Q[1]], 'k--')  # O to Q
plt.plot(tangent_line[:, 0], tangent_line[:, 1], 'm--', label='Tangent at P')
plt.plot(circle_x, circle_y, 'g--', label='Osculating Circle')
plt.plot(C[0], C[1], 'co', markersize=8, label='C (Centre)')
plt.plot([R1[0], R2[0]], [R1[1], R2[1]], 'r-', linewidth=2, label='Chord through O')
plt.plot([M[0], N_pt[0]], [M[1], N_pt[1]], 'orange', linewidth=2, label='Horizontal Chord through P')

# Additional connecting segments:
# RQ: join one end of the chord (chosen here as R1) and Q.
plt.plot([R1[0], Q[0]], [R1[1], Q[1]], 'c--', label='RQ')
# OX and OY: from O to the projections of Q onto the axes.
plt.plot([0, OX[0]], [0, OX[1]], 'y--', label='OX')
plt.plot([0, OY[0]], [0, OY[1]], 'y--', label='OY')
# PM and PN: from P to the endpoints (M and N) of the horizontal chord.
plt.plot([P[0], M[0]], [P[1], M[1]], 'k:', label='PM')
plt.plot([P[0], N_pt[0]], [P[1], N_pt[1]], 'k:', label='PN')

plt.plot(0, 0, 'ks', markersize=10, label='O (Origin)')
plt.axis('equal')
plt.title('Combined Geometric Construction for a Tangential Polar Curve')
plt.legend(loc='best', fontsize=8)
plt.grid(True)
plt.show()
