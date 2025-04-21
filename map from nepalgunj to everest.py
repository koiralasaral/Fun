import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import splprep, splev

# ------------------------------
# A. Define the Route Waypoints
# ------------------------------
# Coordinates: [latitude, longitude]
# Starting at Nepalgunj, then via Chitwan and Kathmandu, to Mount Everest.
waypoints = np.array([
    [28.05, 81.60],        # Nepalgunj (Origin)
    [27.53, 84.35],        # Chitwan
    [27.7172, 85.3240],     # Kathmandu
    [27.9881, 86.9250]      # Mount Everest
])
lats = waypoints[:, 0]
lons = waypoints[:, 1]

# ------------------------------
# B. Spline Interpolation of the Route
# ------------------------------
tck, u = splprep([lons, lats], s=0, k=3)
u_new = np.linspace(0, 1, 200)
lon_smooth, lat_smooth = splev(u_new, tck)
# First derivatives (for tangent estimates)
dlon, dlat = splev(u_new, tck, der=1)
# Second derivatives (for curvature estimation)
d2lon, d2lat = splev(u_new, tck, der=2)

# ------------------------------
# C. Define Local Coordinate System
# ------------------------------
# Use Nepalgunj as the origin O.
origin = np.array([28.05, 81.60])  # [lat, lon]
O_lat, O_lon = origin
# Local coordinates: x = (global lon - O_lon), y = (global lat - O_lat)
x_local = lon_smooth - O_lon
y_local = lat_smooth - O_lat
r_arr = np.sqrt(x_local**2 + y_local**2)  # OP = r for each point

# ------------------------------
# D. Compute Tangents and Pedal Distance
# ------------------------------
# Construct tangent vectors from the derivatives.
# (Since local differences equal global differences in our simple system.)
T_arr = np.vstack((dlon, dlat)).T  
T_norm = np.linalg.norm(T_arr, axis=1)
T_unit = (T_arr.T / T_norm).T  # Unit tangents

# Pedal distance: projection length p = OQ = |(P · T)| for each point.
p_arr = []
for i in range(len(r_arr)):
    P_i = np.array([x_local[i], y_local[i]])
    T_i = T_unit[i]
    p_arr.append(np.abs(np.dot(P_i, T_i)))
p_arr = np.array(p_arr)

# --------------------------------------------
# E. Compute Additional Geometry at a Sample Point
# --------------------------------------------
i_sample = 100  # choose one sample index (approximately midway)
P_sample = np.array([x_local[i_sample], y_local[i_sample]])   # local coordinates of P
r_sample = r_arr[i_sample]  # radial distance
T_sample = T_unit[i_sample]  # unit tangent at P
p_sample = p_arr[i_sample]   # pedal distance
# Foot Q of O onto tangent: Q = (P · T)*T
Q_sample = np.dot(P_sample, T_sample) * T_sample

# Finite difference approximation for dr/dp:
if i_sample > 0 and i_sample < len(r_arr)-1:
    dr = r_arr[i_sample+1] - r_arr[i_sample-1]
    dp = p_arr[i_sample+1] - p_arr[i_sample-1]
    dr_dp = dr / dp if dp != 0 else np.nan
else:
    dr_dp = np.nan
rho_sample = r_sample * dr_dp          # ρ = r·(dr/dp)
PR_sample = 2 * p_sample * dr_dp         # PR = 2p·(dr/dp)

# Angle ψ between OP and the tangent T at P.
psi = np.arccos(np.dot(P_sample, T_sample) / (r_sample + 1e-9))
psi_deg = np.degrees(psi)

# ------------------------------
# F. Compute Osculating Circle and Chord
# ------------------------------
# Curvature k = |x'y'' - y'x''| / ( (x'^2+y'^2)^(3/2) )
num = np.abs(dlon*d2lat - dlat*d2lon)
den = (dlon**2 + dlat**2)**1.5 + 1e-6
k_arr = num / den
k_sample = k_arr[i_sample]
R0 = 1 / k_sample if k_sample != 0 else np.inf

# Unit normal at sample (rotate T_sample 90° counterclockwise)
N_sample = np.array([-T_sample[1], T_sample[0]])
C_local = P_sample + R0 * N_sample  # Centre of osculating circle

# Horizontal chord through P on the osculating circle:
# Solve (x - Cx)^2 + (P_y - Cy)^2 = R0^2  i.e., x = Cx ± sqrt(R0^2 - (P_y - C_y)^2)
disc = R0**2 - (P_sample[1] - C_local[1])**2
if disc >= 0:
    chord_x1 = C_local[0] + np.sqrt(disc)
    chord_x2 = C_local[0] - np.sqrt(disc)
    chord_pts = np.array([[chord_x1, P_sample[1]], [chord_x2, P_sample[1]]])
else:
    chord_pts = np.array([P_sample, P_sample])

# ------------------------------
# G. Matplotlib Animation: Draw the Cumulative Route & Static Geometry
# ------------------------------
fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(np.min(x_local)-0.05, np.max(x_local)+0.05)
ax.set_ylim(np.min(y_local)-0.05, np.max(y_local)+0.05)
ax.set_aspect('equal')
ax.set_title("Matplotlib Animation: Journey from Nepalgunj with Geometry")

# Draw static geometric elements (which remain fixed):
# Radial line (OP) from origin to P_sample:
ax.plot([0, P_sample[0]], [0, P_sample[1]], 'k-', lw=2, label='OP = r')
ax.text(P_sample[0]*0.5, P_sample[1]*0.5, " r", color='k')
# Pedal (OQ):
ax.plot([0, Q_sample[0]], [0, Q_sample[1]], color='purple', lw=2, label='OQ = p')
ax.text(Q_sample[0], Q_sample[1], " Q", color='purple')
# Tangent segment at P_sample:
seg = 0.03
tang_start = P_sample - seg * T_sample
tang_end   = P_sample + seg * T_sample
ax.plot([tang_start[0], tang_end[0]], [tang_start[1], tang_end[1]], 'magenta', lw=2, label='Tangent at P')
ax.text(tang_end[0], tang_end[1], " T", color='magenta')
# Osculating circle:
theta_circ = np.linspace(0, 2*np.pi, 100)
circle_x = C_local[0] + R0 * np.cos(theta_circ)
circle_y = C_local[1] + R0 * np.sin(theta_circ)
ax.plot(circle_x, circle_y, 'g--', lw=2, label='Osculating Circle')
# Chord (horizontal) through P on the circle:
ax.plot(chord_pts[:,0], chord_pts[:,1], 'orange', lw=2, label='Chord (PM)')
ax.text(np.mean(chord_pts[:,0]), chord_pts[0,1], " PM", color='orange')
# Annotate angle ψ at P_sample:
ax.text(P_sample[0], P_sample[1], f"\nψ={psi:.2f} rad\n({psi_deg:.1f}°)", color='red', fontsize=10)

# Prepare the animated cumulative route (the growing polyline)
route_line, = ax.plot([], [], 'b-', lw=3, label='Route')

def init():
    route_line.set_data([], [])
    return route_line,

def animate(i):
    route_line.set_data(x_local[:i+1], y_local[:i+1])
    return route_line,

ani = FuncAnimation(fig, animate, frames=len(u_new), init_func=init,
                    interval=50, blit=True, repeat=True)

ax.legend(loc='upper left')
plt.show()
