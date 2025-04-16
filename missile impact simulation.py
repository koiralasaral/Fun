import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
G = 9.81  # gravity, m/s^2
rho_air = 1.225  # air density, kg/m^3

# Terrain types with density modifiers
terrain_types = {
    "rock": 1.0,
    "sand": 0.8,
    "soil": 0.6,
    "ice": 0.5,
    "water": 0.3
}

# Input Parameters
mass = 1000  # kg
velocity = 3000  # m/s
diameter = 0.5  # m
angle_deg = 45  # degrees
terrain = "rock"

# Derived values
angle_rad = np.radians(angle_deg)
area = np.pi * (diameter / 2) ** 2  # cross-sectional area
ke = 0.5 * mass * velocity ** 2

# Velocity components
v_horizontal = velocity * np.cos(angle_rad)
v_vertical = velocity * np.sin(angle_rad)

# Terrain density modifier
terrain_modifier = terrain_types.get(terrain, 1.0)

# Crater radius (simplified model)
def crater_radius(ke, modifier):
    return (ke / (1e6 * modifier)) ** (1 / 4)  # rough empirical formula

# Penetration depth model (simplified)
def penetration_depth(mass, velocity, angle_rad, terrain_modifier):
    return (mass ** 0.3) * (velocity ** 0.6) * (np.sin(angle_rad)) * 0.01 / terrain_modifier

# Blast radius (scaled down for airbursts or shallow impacts)
def blast_radius(ke):
    return (ke / 4.184e12) ** (1 / 3) * 1000  # Convert J to kilotons TNT equivalent

# Compute values
crater_r = crater_radius(ke, terrain_modifier)
depth = penetration_depth(mass, velocity, angle_rad, terrain_modifier)
blast_r = blast_radius(ke)

# Impact Regime Classification
if velocity < 500:
    regime = "Low energy - likely penetration"
elif velocity < 1500:
    regime = "Moderate energy - mix of penetration and explosion"
else:
    regime = "High energy - explosive impact dominant"

# Print Results
print(f"Missile impact simulation on {terrain} terrain at {angle_deg} degrees:")
print(f"- Kinetic Energy: {ke:.2e} J")
print(f"- Crater Radius: {crater_r:.2f} m")
print(f"- Penetration Depth: {depth:.2f} m")
print(f"- Blast Radius: {blast_r:.2f} m")
print(f"- Impact Regime: {regime}")

# 3D Visualization of crater
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Crater surface
x = np.linspace(-crater_r, crater_r, 100)
y = np.linspace(-crater_r, crater_r, 100)
x, y = np.meshgrid(x, y)
r = np.sqrt(x ** 2 + y ** 2)
z = -depth * (1 - r / crater_r) * (r < crater_r)

ax.plot_surface(x, y, z, cmap='plasma', edgecolor='k', linewidth=0.1, alpha=0.8)
ax.set_title("Simulated Crater Profile")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Depth (m)")
ax.set_box_aspect([1, 1, 0.3])

plt.tight_layout()
plt.show()
