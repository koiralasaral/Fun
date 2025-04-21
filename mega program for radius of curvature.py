import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# ----------------- Symbolic Definitions -----------------
# Define symbolic variable and the pedal (tangential) function.
theta = sp.symbols('theta', real=True)
p_expr = 1 + 0.5 * sp.cos(theta)           # p(θ)
dp_expr = sp.diff(p_expr, theta)             # p′(θ)
# r = √[p(θ)² + (p′(θ))²] is the distance from O to P.
r_expr = sp.sqrt(p_expr**2 + dp_expr**2)

# Cartesian coordinates for the curve: P = (r cosθ, r sinθ)
x_expr = r_expr * sp.cos(theta)
y_expr = r_expr * sp.sin(theta)

# Create numeric functions for evaluation.
x_func = sp.lambdify(theta, x_expr, 'numpy')
y_func = sp.lambdify(theta, y_expr, 'numpy')
r_func = sp.lambdify(theta, r_expr, 'numpy')

# Compute derivatives for the tangent.
dx_expr = sp.diff(x_expr, theta)
dy_expr = sp.diff(y_expr, theta)
dx_func = sp.lambdify(theta, dx_expr, 'numpy')
dy_func = sp.lambdify(theta, dy_expr, 'numpy')

# For the curvature we use the Cartesian formula:
d2x_expr = sp.diff(dx_expr, theta)
d2y_expr = sp.diff(dy_expr, theta)
curvature_expr = sp.Abs(dx_expr * d2y_expr - dy_expr * d2x_expr) / (dx_expr**2 + dy_expr**2)**(sp.Rational(3, 2))
R_curvature_expr = 1 / curvature_expr  # R0 = 1/κ : radius of curvature
R_curvature_func = sp.lambdify(theta, R_curvature_expr, 'numpy')

# ----------------- Choose a Specific Point -----------------
# Let θ₀ = π/4 be our selected angle.
theta0 = np.pi/4
P = np.array([x_func(theta0), y_func(theta0)])  # point P on the curve, in Cartesian coordinates.
r_val = np.linalg.norm(P)                        # OP = r
# Compute the pedal (p = OQ) at θ₀:
p_val = 1 + 0.5 * np.cos(theta0)

# ----------------- Tangent and Related Constructions -----------------
# Compute the derivative at θ₀; the tangent vector is:
dP = np.array([dx_func(theta0), dy_func(theta0)])
T_vec = dP / np.linalg.norm(dP)    # Unit tangent vector at P.
# The foot Q of the perpendicular from O onto the tangent:
Q = (np.dot(P, T_vec)) * T_vec     # OQ = p.

# Compute the angle of the tangent:
angle_T = np.arctan2(T_vec[1], T_vec[0])
# By convention, the polar angle of P is θ₀.
# The angle between OP and the tangent is then ψ:
psi_val = np.abs(angle_T - theta0)  # Alternatively, note that sin(ψ) = p/r.

# ----------------- Osculating Circle -----------------
# Radius of curvature at θ₀:
R0 = R_curvature_func(theta0)
# The normal vector N is obtained by rotating T by +90°:
N_vec = np.array([-T_vec[1], T_vec[0]])
# Centre of osculating circle: C = P + R0·N.
C = P + R0 * N_vec

# Define φ (phi) as the angle that the vector CP (from C to P) makes with the x-axis.
CP = P - C
phi_val = np.arctan2(CP[1], CP[0])

# ----------------- Additional Tangent Intersections -----------------
# (a) Intersection of the tangent with the osculating circle (other than P).
# Since P lies on the circle we have:
#   ||(P + s·T_vec) - C||² = R0². One solution is s = 0; the other is:
s_int = -2 * np.dot(P - C, T_vec)
T_int = P + s_int * T_vec  # The second tangent–circle intersection; label it T.

# (b) Intersection of the tangent with the x–axis.
if np.abs(T_vec[1]) > 1e-6:
    s_x = - P[1] / T_vec[1]
    X_point = P + s_x * T_vec
else:
    X_point = None

# ----------------- Axis–Parallel Chords -----------------
# Horizontal chord (parallel to the x–axis) through the line y = P[1]:
disc_h = R0**2 - (P[1] - C[1])**2
if disc_h >= 0:
    sqrt_disc_h = np.sqrt(disc_h)
    M1 = np.array([C[0] + sqrt_disc_h, P[1]])
    M2 = np.array([C[0] - sqrt_disc_h, P[1]])
else:
    M1 = M2 = P

# Vertical chord (parallel to the y–axis) through the line x = P[0]:
disc_v = R0**2 - (P[0] - C[0])**2
if disc_v >= 0:
    sqrt_disc_v = np.sqrt(disc_v)
    N1 = np.array([P[0], C[1] + sqrt_disc_v])
    N2 = np.array([P[0], C[1] - sqrt_disc_v])
else:
    N1 = N2 = P

# ----------------- Plotting and Annotations -----------------
plt.figure(figsize=(12, 12))
ax = plt.gca()

# Plot the full curve for context:
theta_vals = np.linspace(0, 2*np.pi, 400)
curve_x = x_func(theta_vals)
curve_y = y_func(theta_vals)
plt.plot(curve_x, curve_y, 'b-', label='Curve P(r,θ)')

# Plot point P and its radial line from O:
plt.plot(P[0], P[1], 'ro', label='P (r,θ)')
plt.text(P[0], P[1], '  P', color='r')
plt.plot([0, P[0]], [0, P[1]], 'k-', label='OP = r')
mid_OP = 0.5 * P
plt.text(mid_OP[0], mid_OP[1], '  r', color='k')

# Draw an arc at O indicating angle θ (from the x–axis to OP).
theta_deg = np.degrees(theta0)
arc_theta = Arc((0, 0), 0.4 * r_val, 0.4 * r_val, angle=0, theta1=0, theta2=theta_deg, color='purple')
ax.add_patch(arc_theta)
plt.text(0.4 * r_val * np.cos(theta0/2), 0.4 * r_val * np.sin(theta0/2), r'$\theta$', color='purple')

# Plot the tangent line at P.
s_vals = np.linspace(-1, 1, 100)
tangent_line = np.array([P + s * T_vec for s in s_vals])
plt.plot(tangent_line[:, 0], tangent_line[:, 1], 'm--', label='Tangent at P')

# Mark the foot Q and draw OQ:
plt.plot(Q[0], Q[1], 'ko', label='Q (foot from O to tangent)')
plt.text(Q[0], Q[1], '  Q', color='k')
plt.plot([0, Q[0]], [0, Q[1]], 'k:', label='OQ = p')
mid_OQ = 0.5 * Q
plt.text(mid_OQ[0], mid_OQ[1], '  p', color='k')

# Draw the osculating circle.
phi_vals = np.linspace(0, 2*np.pi, 200)
circle_x = C[0] + R0 * np.cos(phi_vals)
circle_y = C[1] + R0 * np.sin(phi_vals)
plt.plot(circle_x, circle_y, 'g--', label='Osculating Circle')
plt.plot(C[0], C[1], 'co', label='C (Centre)')
plt.text(C[0], C[1], '  C', color='c')
# Draw line from P to C.
plt.plot([P[0], C[0]], [P[1], C[1]], 'c:', label='PC (radius of circle)')

# Mark the second intersection of the tangent with the circle, point T.
plt.plot(T_int[0], T_int[1], 'bo', label='T (second intersection)')
plt.text(T_int[0], T_int[1], '  T', color='b')
# Draw segments along the tangent:
# (a) QT: from Q to T.
plt.plot([Q[0], T_int[0]], [Q[1], T_int[1]], 'c--', label='QT')
# (b) TX: from T to intersection X with x–axis.
if X_point is not None:
    plt.plot(X_point[0], X_point[1], 'yo', label='X (tangent–x-axis)')
    plt.text(X_point[0], X_point[1], '  X', color='y')
    plt.plot([T_int[0], X_point[0]], [T_int[1], X_point[1]], 'y--', label='TX')

# Draw and label the angle ψ at P (between OP and tangent).
# We draw a small arc centered at P.
psi_deg = np.degrees(psi_val)
arc_psi = Arc(P, 0.4 * r_val, 0.4 * r_val, angle=np.degrees(theta0), theta1=0, theta2=psi_deg, color='red')
ax.add_patch(arc_psi)
plt.text(P[0] + 0.3 * r_val * np.cos(theta0 + np.radians(psi_deg)/2),
         P[1] + 0.3 * r_val * np.sin(theta0 + np.radians(psi_deg)/2),
         r'$\psi$', color='red')

# Draw and label the angle φ at C (between CP and the x–axis).
arc_phi = Arc(C, 0.3 * R0, 0.3 * R0, angle=0, theta1=0, theta2=np.degrees(phi_val), color='magenta')
ax.add_patch(arc_phi)
plt.text(C[0] + 0.35 * R0 * np.cos(phi_val/2),
         C[1] + 0.35 * R0 * np.sin(phi_val/2),
         r'$\phi$', color='magenta')

# Draw the horizontal chord through the osculating circle (parallel to the x–axis).
plt.plot([M1[0], M2[0]], [M1[1], M2[1]], 'orange', linewidth=2, label='Chord PM (∥ x-axis)')
plt.text(0.5*(M1[0] + M2[0]), M1[1], '  PM', color='orange')

# Draw the vertical chord through the osculating circle (parallel to the y–axis).
plt.plot([N1[0], N2[0]], [N1[1], N2[1]], 'brown', linewidth=2, label='Chord PN (∥ y-axis)')
plt.text(N1[0], 0.5*(N1[1] + N2[1]), '  PN', color='brown')

# Draw the coordinate axes (OX and OY) for reference.
x_axis = np.linspace(-r_val, 1.5*r_val, 100)
plt.plot(x_axis, np.zeros_like(x_axis), 'k-', alpha=0.3, label='x-axis')
y_axis = np.linspace(-r_val, 1.5*r_val, 100)
plt.plot(np.zeros_like(y_axis), y_axis, 'k-', alpha=0.3, label='y-axis')

plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comprehensive Geometric Construction for a Pedal Curve\nwith OP = r, OQ = p, Tangent (QT, TX), Angles θ, ψ, φ, and Chords PM & PN')
plt.legend(loc='best', fontsize=8)
plt.grid(True)
plt.show()
