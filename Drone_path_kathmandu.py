import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# =======================================================
# PART 1: DRONE PATH SMOOTHING (Using Kathmandu Waypoints)
# =======================================================
# Sample waypoints (lat, lon) approximating a flight over Kathmandu.
# Here latitudes and longitudes are given simply as numbers.
waypoints = np.array([
    [27.7048, 85.3200],
    [27.7070, 85.3210],
    [27.7095, 85.3220],
    [27.7120, 85.3228],
    [27.7150, 85.3235],
    [27.7180, 85.3240],
    [27.7210, 85.3245],
    [27.7240, 85.3250]
])

# For interpolation, treat longitude as x and latitude as y.
# Spline interpolation (cubic, with s=0 to interpolate exactly)
lon = waypoints[:,1]
lat = waypoints[:,0]
tck, u = splprep([lon, lat], s=0, k=3)
u_new = np.linspace(0, 1, 100)
lon_smooth, lat_smooth = splev(u_new, tck)
dx, dy = splev(u_new, tck, der=1)
d2x, d2y = splev(u_new, tck, der=2)
# Compute curvature: k = |dx*d2y - dy*d2x| / (dx^2+dy^2)^(3/2)
curvature = np.abs(dx * d2y - dy * d2x) / ( (dx**2 + dy**2)**(1.5) + 1e-6 )

# =======================================================
# PART 2: GEOMETRIC CONSTRUCTION (Annotated for Pedal/Pedal‐like Curve)
# =======================================================
# For our local coordinate system, choose O as the first point on the smooth path.
O_global = np.array([lon_smooth[0], lat_smooth[0]])
# Define local coordinates by subtracting O_global (so that O = (0, 0)).
trajectory = np.vstack((lon_smooth, lat_smooth)).T - O_global

# Choose a representative point along the trajectory. Here we pick index i = 50.
i = 50
P_global = np.array([lon_smooth[i], lat_smooth[i]])
P = P_global - O_global  # Local coordinates for P; now O = (0,0).
r_val = np.linalg.norm(P)  # OP = r (distance from O to P)
theta_val = np.arctan2(P[1], P[0])  # Polar angle θ (in radians)

# Tangent at P (from the spline derivative).
T = np.array([dx[i], dy[i]])
T = T / np.linalg.norm(T)  # Unit tangent vector

# Foot Q: projection of O=(0,0) onto the tangent line at P.
# For a line L(s)= P + s*T, the foot Q minimizes ||P+s*T||, yielding s = - (P dot T).
s_Q = -np.dot(P, T)
Q = P + s_Q * T
p_val = np.linalg.norm(Q)  # p = OQ (the pedal distance)

# Curvature and osculating circle at P.
k_val = curvature[i]
R0 = 1.0 / k_val if k_val != 0 else np.inf   # radius of curvature

# Unit normal vector, defined by rotating T by 90° (counterclockwise).
N = np.array([-T[1], T[0]])
# Centre of the osculating circle: C = P + R0·N.
C = P + R0 * N

# Angle φ: the angle (with the positive x-axis) made by the vector from C to P.
CP = P - C
phi_val = np.arctan2(CP[1], CP[0])

# Angle ψ: the angle between the vector OP and the tangent T.
psi_val = np.arccos(np.dot(T, P)/r_val)

# Intersection of the tangent with the x-axis.
if np.abs(T[1]) > 1e-6:
    s_x = - P[1] / T[1]
    X = P + s_x * T
else:
    X = None

# Horizontal chord on the osculating circle (parallel to x‑axis) through P.
# Solve (x - Cx)^2 + (P_y - C_y)^2 = R0^2.
disc_h = R0**2 - (P[1] - C[1])**2
if disc_h >= 0:
    sqrt_disc_h = np.sqrt(disc_h)
    M1 = np.array([C[0] + sqrt_disc_h, P[1]])
    M2 = np.array([C[0] - sqrt_disc_h, P[1]])
else:
    M1 = M2 = P

# Vertical chord on the osculating circle (parallel to y‑axis) through P.
disc_v = R0**2 - (P[0] - C[0])**2
if disc_v >= 0:
    sqrt_disc_v = np.sqrt(disc_v)
    N1 = np.array([P[0], C[1] + sqrt_disc_v])
    N2 = np.array([P[0], C[1] - sqrt_disc_v])
else:
    N1 = N2 = P

# =======================================================
# PART 3: VISUALIZATION WITH ANNOTATIONS
# =======================================================
plt.figure(figsize=(10,10))

# Plot the entire local (translated) trajectory.
plt.plot(trajectory[:,0], trajectory[:,1], 'b-', label='Smoothed trajectory')
# Mark point P.
plt.plot(P[0], P[1], 'ro', label='P (r,θ)')
plt.text(P[0], P[1], '  P', color='r')
# Draw the radial line OP.
plt.plot([0, P[0]], [0, P[1]], 'k-', label='OP = r')
mid_OP = 0.5 * P
plt.text(mid_OP[0], mid_OP[1], '  r', color='k')

# Label the polar angle θ at O (draw a short arc or simply annotate).
plt.annotate(r'$\theta$', xy=(0.2*r_val*np.cos(theta_val), 0.2*r_val*np.sin(theta_val)),
             fontsize=12, color='purple')

# Plot the tangent line at P.
s_vals = np.linspace(-1, 1, 100)
tangent_line = np.array([P + s * T for s in s_vals])
plt.plot(tangent_line[:,0], tangent_line[:,1], 'm--', label='Tangent at P')

# Mark the foot Q and draw OQ (pedal).
plt.plot(Q[0], Q[1], 'ko', label='Q (OQ = p)')
plt.text(Q[0], Q[1], '  Q', color='k')
plt.plot([0, Q[0]], [0, Q[1]], 'k:', label='OQ = p')

# Plot the osculating circle.
phi_circle = np.linspace(0, 2*np.pi, 200)
circle_x = C[0] + R0 * np.cos(phi_circle)
circle_y = C[1] + R0 * np.sin(phi_circle)
plt.plot(circle_x, circle_y, 'g--', label='Osculating Circle')
plt.plot(C[0], C[1], 'co', label='C (Centre)')
plt.text(C[0], C[1], '  C', color='c')
# Draw line PC (radius of the circle).
plt.plot([P[0], C[0]], [P[1], C[1]], 'c:', label='PC (R)')
plt.text((P[0]+C[0])/2, (P[1]+C[1])/2, ' R', color='c')

# Annotate angle ψ (between OP and the tangent) at P.
plt.annotate(r'$\psi$', xy=(P[0] + 0.2*r_val*np.cos(theta_val + psi_val/2),
                             P[1] + 0.2*r_val*np.sin(theta_val + psi_val/2)),
             fontsize=12, color='red')

# Annotate angle φ at C (angle that CP makes with the x-axis).
plt.annotate(r'$\phi$', xy=(C[0] + 0.2*R0*np.cos(phi_val/2),
                             C[1] + 0.2*R0*np.sin(phi_val/2)),
             fontsize=12, color='magenta')

# Draw the intersection X: tangent with x-axis.
if X is not None:
    plt.plot(X[0], X[1], 'yo', label='X (tangent - x-axis)')
    plt.text(X[0], X[1], '  X', color='y')
    plt.plot([P[0], X[0]], [P[1], X[1]], 'y--', label='TX')

# Draw horizontal chord (PM) through P on the osculating circle.
plt.plot([M1[0], M2[0]], [M1[1], M2[1]], 'orange', linewidth=2, label='Chord PM ∥ x-axis')
plt.text(0.5*(M1[0]+M2[0]), M1[1], '  PM', color='orange')

# Draw vertical chord (PN) through P on the osculating circle.
plt.plot([N1[0], N2[0]], [N1[1], N2[1]], 'brown', linewidth=2, label='Chord PN ∥ y-axis')
plt.text(N1[0], 0.5*(N1[1]+N2[1]), '  PN', color='brown')

# Draw coordinate axes for reference.
x_axis = np.linspace(-r_val*1.5, r_val*1.5, 100)
plt.plot(x_axis, np.zeros_like(x_axis), 'k-', alpha=0.3, label='x-axis')
y_axis = np.linspace(-r_val*1.5, r_val*1.5, 100)
plt.plot(np.zeros_like(y_axis), y_axis, 'k-', alpha=0.3, label='y-axis')

plt.axis('equal')
plt.xlabel('x (local units)')
plt.ylabel('y (local units)')
plt.title('Drone Trajectory -- Annotated Geometry\nShowing OP = r, OQ = p, Tangent (with ψ),\nOsculating Circle (with φ) & Chords PM, PN; θ is at O')
plt.legend(loc='best', fontsize=9)
plt.grid(True)
plt.show()

# =======================================================
# PART 4: PRINT INTERMEDIATE VALUES
# =======================================================
print("Intermediate Values:")
print(f"Selected point P (local coordinates): {P}")
print(f"OP = r: {r_val:.4f} (local units)")
print(f"Angle θ (radians): {theta_val:.4f} ({np.degrees(theta_val):.2f}°)")
print(f"Foot Q (OQ = p): {Q}, p = {p_val:.4f}")
print(f"Unit Tangent T: {T}")
print(f"Curvature k: {k_val:.4f}, R0: {R0:.4f} (osculating circle radius)")
print(f"Centre C of osculating circle: {C}")
print(f"Angle ψ (radians): {psi_val:.4f} ({np.degrees(psi_val):.2f}°)")
print(f"Angle φ (radians): {phi_val:.4f} ({np.degrees(phi_val):.2f}°)")
if X is not None:
    print(f"Intersection X (tangent with x-axis): {X}")
print(f"Horizontal chord endpoints (PM): {M1} and {M2}")
print(f"Vertical chord endpoints (PN): {N1} and {N2}")
