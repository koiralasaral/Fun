import numpy as np
import matplotlib.pyplot as plt

# --- Define Bridge Parameters ---
L = 50.0              # Total span of the bridge in meters.
q = 20000.0           # Uniform load (N/m), representing the distributed traffic load.
# Assume the beam's cross-sectional properties (for a typical reinforced concrete or steel girder):
I = 0.1               # Moment of inertia in m^4 (assumed value; adjust with real data).
c = 0.5               # Distance from the neutral axis to the extreme fiber in m.

# --- Compute Bending Moment Distribution ---
# For a simply supported beam under uniform load, bending moment at x: M(x) = q*x*(L-x)/2.
x_vals = np.linspace(0, L, 100)
M_vals = q * x_vals * (L - x_vals) / 2

# --- Compute Bending Stress ---
# Bending stress sigma = M * c / I (in N/m^2 or Pascals)
stress_vals = M_vals * c / I

# Print intermediate values at key positions: x=0, L/2, L.
print("=== Bridge Stress Analysis Intermediate Values ===")
for x in [0, L/2, L]:
    M = q * x * (L - x) / 2
    sigma = M * c / I
    print(f"x = {x:.2f} m: Bending Moment M = {M:.2f} N·m, Bending Stress = {sigma:.2f} Pa")

# --- Plot the Bending Moment Distribution ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_vals, M_vals, 'b-', linewidth=2)
plt.xlabel('Position along span (m)')
plt.ylabel('Bending Moment (N·m)')
plt.title('Bending Moment Distribution along the Bridge Span')
plt.grid(True)

# --- Plot the Bending Stress Distribution ---
plt.subplot(1, 2, 2)
plt.plot(x_vals, stress_vals, 'r-', linewidth=2)
plt.xlabel('Position along span (m)')
plt.ylabel('Bending Stress (Pa)')
plt.title('Bending Stress Distribution across the Bridge Span')
plt.grid(True)
plt.tight_layout()
plt.show()
