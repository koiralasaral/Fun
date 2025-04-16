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
    "water": 0.3,
    "lead": 1.5
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
# Add missile specifications
missile_specs = {
    "HQ-9": {
        "range_km": 200,
        "max_altitude_km": 30,
        "speed_mach": 4.2,
        "warhead_kg": 180,
        "engagement_time_s": 120,
        "cost_million": 1.2,
        "pk": 0.7,
        "mtbf_hours": 500,
        "repair_time_hours": 50,
        "radar_range": 300
    },
    "Patriot PAC-3": {
        "range_km": 70,
        "max_altitude_km": 24,
        "speed_mach": 5,
        "warhead_kg": 74,
        "engagement_time_s": 90,
        "cost_million": 4,
        "pk": 0.85,
        "mtbf_hours": 700,
        "repair_time_hours": 30,
        "radar_range": 150
    },
    "9K720 ISKANDER-M": {
        "range_km": 500,
        "max_altitude_km": 50,
        "speed_mach": 6.0,
        "warhead_kg": 480,
        "engagement_time_s": 180,
        "cost_million": 3.5,
        "pk": 0.9,
        "mtbf_hours": 1000,
        "repair_time_hours": 40,
        "radar_range": 400
    },
    "ATACMS": {
        "range_km": 300,
        "max_altitude_km": 50,
        "speed_mach": 3.5,
        "warhead_kg": 230,
        "engagement_time_s": 150,
        "cost_million": 2.5,
        "pk": 0.8,
        "mtbf_hours": 800,
        "repair_time_hours": 35,
        "radar_range": 350
    }
}

# Add subplots for missile specifications
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Range vs Cost
ranges = [spec["range_km"] for spec in missile_specs.values()]
costs = [spec["cost_million"] for spec in missile_specs.values()]
axs[0, 0].bar(missile_specs.keys(), ranges, color='blue', alpha=0.7, label='Range (km)')
axs[0, 0].set_ylabel("Range (km)")
axs[0, 0].set_title("Missile Range vs Cost")
axs[0, 0].tick_params(axis='x', rotation=45)
axs[0, 0].legend(loc="upper left")

ax2 = axs[0, 0].twinx()
ax2.plot(missile_specs.keys(), costs, color='red', marker='o', label='Cost (Million $)')
ax2.set_ylabel("Cost (Million $)")
ax2.legend(loc="upper right")

# Plot 2: Speed vs Warhead Weight
speeds = [spec["speed_mach"] for spec in missile_specs.values()]
warheads = [spec["warhead_kg"] for spec in missile_specs.values()]
axs[0, 1].bar(missile_specs.keys(), speeds, color='green', alpha=0.7, label='Speed (Mach)')
axs[0, 1].set_ylabel("Speed (Mach)")
axs[0, 1].set_title("Missile Speed vs Warhead Weight")
axs[0, 1].tick_params(axis='x', rotation=45)
axs[0, 1].legend(loc="upper left")

ax3 = axs[0, 1].twinx()
ax3.plot(missile_specs.keys(), warheads, color='orange', marker='o', label='Warhead Weight (kg)')
ax3.set_ylabel("Warhead Weight (kg)")
ax3.legend(loc="upper right")

# Plot 3: Probability of Kill (Pk) vs Radar Range
pks = [spec["pk"] for spec in missile_specs.values()]
radar_ranges = [spec["radar_range"] for spec in missile_specs.values()]
axs[1, 0].bar(missile_specs.keys(), pks, color='purple', alpha=0.7, label='Pk')
axs[1, 0].set_ylabel("Probability of Kill (Pk)")
axs[1, 0].set_title("Missile Pk vs Radar Range")
axs[1, 0].tick_params(axis='x', rotation=45)
axs[1, 0].legend(loc="upper left")

ax4 = axs[1, 0].twinx()
ax4.plot(missile_specs.keys(), radar_ranges, color='brown', marker='o', label='Radar Range (km)')
ax4.set_ylabel("Radar Range (km)")
ax4.legend(loc="upper right")

# Plot 4: MTBF vs Repair Time
mtbf = [spec["mtbf_hours"] for spec in missile_specs.values()]
repair_times = [spec["repair_time_hours"] for spec in missile_specs.values()]
axs[1, 1].bar(missile_specs.keys(), mtbf, color='cyan', alpha=0.7, label='MTBF (hours)')
axs[1, 1].set_ylabel("MTBF (hours)")
axs[1, 1].set_title("Missile MTBF vs Repair Time")
axs[1, 1].tick_params(axis='x', rotation=45)
axs[1, 1].legend(loc="upper left")

ax5 = axs[1, 1].twinx()
ax5.plot(missile_specs.keys(), repair_times, color='magenta', marker='o', label='Repair Time (hours)')
ax5.set_ylabel("Repair Time (hours)")
ax5.legend(loc="upper right")

plt.tight_layout()
plt.show()

# Show crater for all missiles
for missile, spec in missile_specs.items():
    ke = 0.5 * spec["warhead_kg"] * (spec["speed_mach"] * 343) ** 2  # Speed in m/s
    crater_r = crater_radius(ke, terrain_modifier)
    depth = penetration_depth(spec["warhead_kg"], spec["speed_mach"] * 343, angle_rad, terrain_modifier)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Crater surface
    x = np.linspace(-crater_r, crater_r, 100)
    y = np.linspace(-crater_r, crater_r, 100)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x ** 2 + y ** 2)
    z = -depth * (1 - r / crater_r) * (r < crater_r)

    ax.plot_surface(x, y, z, cmap='plasma', edgecolor='k', linewidth=0.1, alpha=0.8)
    ax.set_title(f"Simulated Crater Profile for {missile}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")
    ax.set_box_aspect([1, 1, 0.3])

    plt.tight_layout()
    plt.show()