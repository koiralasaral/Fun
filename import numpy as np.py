import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

# Enhanced missile specifications
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
    }
}

class AirDefenseAnalyzer:
    def __init__(self, specs):
        self.specs = specs
        
    def plot_trajectories(self, target_distance, target_altitude):
        plt.figure(figsize=(10,6))
        for system in self.specs:
            if target_distance <= self.specs[system]["range_km"] and target_altitude <= self.specs[system]["max_altitude_km"]:
                t = np.linspace(0, 1, 100)
                x = target_distance * t
                y = 4 * target_altitude * (t - t**2)
                plt.plot(x, y, label=system)
        plt.title(f"Missile Trajectories to {target_distance}km/{target_altitude}km Target")
        plt.xlabel("Distance (km)"); plt.ylabel("Altitude (km)")
        plt.legend(); plt.grid(True)
        plt.show()
    
    def probability_of_kill(self, n_interceptors):
        plt.figure(figsize=(10,6))
        for system in self.specs:
            pk = [1 - (1 - self.specs[system]["pk"])**n for n in range(1, n_interceptors+1)]
            plt.plot(range(1, n_interceptors+1), pk, 'o-', label=system)
        plt.title("Cumulative Probability of Kill")
        plt.xlabel("Number of Interceptors"); plt.ylabel("Pₖ")
        plt.legend(); plt.grid(True)
        plt.show()
    
    def lanchester_attrition(self, hq9_count, patriot_count, duration=10):
        def model(y, t, alpha, beta):
            hq9, patriot = y
            dhq9 = -beta * patriot
            dpatriot = -alpha * hq9
            return [dhq9, dpatriot]
        
        alpha = self.specs["HQ-9"]["pk"] * 0.4  # Effectiveness factors
        beta = self.specs["Patriot PAC-3"]["pk"] * 0.6
        t = np.linspace(0, duration, 100)
        solution = odeint(model, [hq9_count, patriot_count], t, args=(alpha, beta))
        
        plt.figure(figsize=(10,6))
        plt.plot(t, solution[:,0], label='HQ-9 Forces')
        plt.plot(t, solution[:,1], label='Patriot Forces')
        plt.title("Force Attrition Over Time")
        plt.xlabel("Time (arbitrary units)"); plt.ylabel("Remaining Units")
        plt.legend(); plt.grid(True)
        plt.show()
        
        final_ratio = solution[-1,0] / solution[-1,1]
        print(f"Final force ratio: HQ-9/Patriot = {final_ratio:.2f}:1")
    
    def monte_carlo_battle(self, hq9_count, patriot_count, n_simulations=5000):
        hq9_wins = 0
        for _ in range(n_simulations):
            hq9, patriot = hq9_count, patriot_count
            while hq9 > 0 and patriot > 0:
                hq9 -= np.random.binomial(max(0, patriot), self.specs["Patriot PAC-3"]["pk"]*0.5)
                patriot -= np.random.binomial(max(0, hq9), self.specs["HQ-9"]["pk"]*0.4)
            hq9_wins += 1 if patriot <= 0 else 0
        
        plt.figure(figsize=(8,5))
        plt.bar(["HQ-9 Victory", "Patriot Victory"], 
                [hq9_wins/n_simulations, 1-hq9_wins/n_simulations])
        plt.title(f"Battle Outcome Probability ({hq9_count}v{patriot_count})")
        plt.ylabel("Probability"); plt.ylim(0,1)
        plt.show()
    
    def cost_analysis(self, n_targets):
        costs = {}
        for system in self.specs:
            missiles_needed = np.ceil(n_targets / self.specs[system]["pk"])
            costs[system] = missiles_needed * self.specs[system]["cost_million"]
        
        plt.figure(figsize=(8,5))
        plt.bar(costs.keys(), costs.values())
        plt.title(f"Total Cost to Engage {n_targets} Targets")
        plt.ylabel("Cost ($M)"); plt.grid(True)
        plt.show()
        
        for system, cost in costs.items():
            print(f"{system}: ${cost:.1f}M (${cost/n_targets:.2f}M per target)")
    
    def operational_availability(self, total_units):
        available = {}
        for system in self.specs:
            mtbf = self.specs[system]["mtbf_hours"]
            repair = self.specs[system]["repair_time_hours"]
            available[system] = int(total_units * mtbf/(mtbf + repair))
        
        plt.figure(figsize=(8,5))
        plt.bar(available.keys(), available.values())
        plt.title(f"Operational Units (Maintenance Considered)\nFrom {total_units} Total Units")
        plt.ylabel("Available Units"); plt.grid(True)
        plt.show()
    
    def radar_coverage(self):
        def calc_coverage(range_km, alt_km):
            earth_radius = 6371  # Earth's radius in km
            return 2 * np.pi * earth_radius * (sqrt(alt_km**2 + 2 * earth_radius * alt_km) - alt_km)
        
        coverages = {}
        for system in self.specs:
            coverages[system] = calc_coverage(
                self.specs[system]["radar_range"],
                self.specs[system]["max_altitude_km"])
        
        plt.figure(figsize=(8,5))
        plt.bar(coverages.keys(), [c/1e6 for c in coverages.values()])  # In million km²
        plt.title("Radar Coverage Area")
        plt.ylabel("Coverage Area (million km²)"); plt.grid(True)
        plt.show()

# Execute complete analysis
analyzer = AirDefenseAnalyzer(missile_specs)

print("=== Trajectory Analysis ===")
analyzer.plot_trajectories(150, 20)

print("\n=== Probability of Kill Analysis ===")
analyzer.probability_of_kill(5)

print("\n=== Lanchester Attrition Model ===")
analyzer.lanchester_attrition(20, 15)

print("\n=== Monte Carlo Battle Simulation ===")
analyzer.monte_carlo_battle(20, 15)

print("\n=== Cost Effectiveness Analysis ===")
analyzer.cost_analysis(10)

print("\n=== Operational Availability ===")
analyzer.operational_availability(100)

print("\n=== Radar Coverage Analysis ===")
analyzer.radar_coverage()
print("\n=== Regression Analysis ===")

# Extract data for regression analysis
systems = list(missile_specs.keys())
ranges = [missile_specs[system]["range_km"] for system in systems]
costs = [missile_specs[system]["cost_million"] for system in systems]

# Perform least squares regression
x = np.array(ranges)
y = np.array(costs)
A = np.vstack([x, np.ones(len(x))]).T
coefficients, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
slope, intercept = coefficients

# Predicted values
y_pred = slope * x + intercept

# Mean residual sum of squares
mean_residual_sum_of_squares = np.mean((y - y_pred) ** 2)

# Plot data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Data", color="blue")
plt.plot(x, y_pred, label=f"Fit: y = {slope:.2f}x + {intercept:.2f}", color="red")
plt.title("Linear Regression: Range vs Cost")
plt.xlabel("Range (km)"); plt.ylabel("Cost (Million $)")
plt.legend(); plt.grid(True)
plt.show()

# Residual plot
residuals = y - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(x, residuals, color="purple")
plt.axhline(0, color="black", linestyle="--")
plt.title("Residual Plot: Range vs Cost")
plt.xlabel("Range (km)"); plt.ylabel("Residuals")
plt.grid(True)
plt.show()

# Interpretation
print(f"Slope: {slope:.2f}, Intercept: {intercept:.2f}")
print(f"Mean Residual Sum of Squares: {mean_residual_sum_of_squares:.2f}")
print("Residual plot shows the distribution of errors. Ideally, residuals should be randomly scattered around zero, indicating a good fit.")
# Show trajectory of both systems for comparison
plt.figure(figsize=(10, 6))
t = np.linspace(0, 1, 100)
for system in missile_specs:
    range_km = missile_specs[system]["range_km"]
    max_altitude_km = missile_specs[system]["max_altitude_km"]
    x = range_km * t
    y = 4 * max_altitude_km * (t - t**2)
    plt.plot(x, y, label=system)

plt.title("Missile Trajectories Comparison")
plt.xlabel("Distance (km)")
plt.ylabel("Altitude (km)")
plt.legend()
plt.grid(True)
plt.show()
def impact_radius(missile_mass, velocity, target_density, target_yield_strength):
    """
    Calculate the radius of impact crater using a simplified model.
    """
    kinetic_energy = 0.5 * missile_mass * velocity**2  # Kinetic energy of the missile
    crater_volume = kinetic_energy / target_yield_strength  # Simplified volume estimation
    radius = (3 * crater_volume / (4 * np.pi))**(1/3)  # Convert volume to radius
    return radius
# Parameters for a lighter material
lighter_target_density = 1500  # kg/m³ (e.g., loose soil)
lighter_target_yield_strength = 5e6  # Pa (weaker material)
# Parameters for the calculation
missile_mass = 180  # kg (warhead mass of HQ-9)
velocity = 4.2 * 343  # m/s (Mach 4.2, speed of sound ~343 m/s)
target_density = 2500  # kg/m³ (density of target material, e.g., rock)
target_yield_strength = 1e7  # Pa (yield strength of target material)
    # Calculate radius for lighter material
lighter_radius = impact_radius(missile_mass, velocity, lighter_target_density, lighter_target_yield_strength)
print(f"Estimated impact radius for lighter material: {lighter_radius:.2f} meters")

    # Plot 3D diagram of the impact crater for lighter material
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Create a 3D crater
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
# Calculate radius
radius = impact_radius(missile_mass, velocity, target_density, target_yield_strength)
print(f"Estimated impact radius: {radius:.2f} meters")

x = radius * np.outer(np.cos(u), np.sin(v))
y = radius * np.outer(np.sin(u), np.sin(v))
z = -radius * np.outer(np.ones(np.size(u)), np.cos(v))  # Negative z for crater depth
    # Create a 3D crater for lighter material
x = lighter_radius * np.outer(np.cos(u), np.sin(v))
y = lighter_radius * np.outer(np.sin(u), np.sin(v))
z = -lighter_radius * np.outer(np.ones(np.size(u)), np.cos(v))  # Negative z for crater depth

ax.plot_surface(x, y, z, color='tan', alpha=0.7)
ax.set_title("3D Impact Crater (Lighter Material)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Depth (m)")
plt.show()    


# 
# Plot 3D diagram of the impact crater
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')



ax.plot_surface(x, y, z, color='brown', alpha=0.7)
ax.set_title("3D Impact Crater")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Depth (m)")
plt.show()
# Calculate and display crater radius and depth
crater_depth = radius  # Depth is approximately equal to the radius in this simplified model
print(f"Crater Radius: {radius:.2f} meters")
print(f"Crater Depth: {crater_depth:.2f} meters")

# Compare with asteroid impact
asteroid_mass = 1e15  # kg (estimated mass of the Chicxulub asteroid)
asteroid_velocity = 20000  # m/s (estimated velocity of the Chicxulub asteroid)
asteroid_kinetic_energy = 0.5 * asteroid_mass * asteroid_velocity**2
asteroid_crater_volume = asteroid_kinetic_energy / target_yield_strength
asteroid_crater_radius = (3 * asteroid_crater_volume / (4 * np.pi))**(1/3)

print(f"Estimated Chicxulub Crater Radius: {asteroid_crater_radius / 1000:.2f} km")

# Calculate number of missiles required to match Chicxulub crater
missile_kinetic_energy = 0.5 * missile_mass * velocity**2
missiles_required = asteroid_kinetic_energy / missile_kinetic_energy
print(f"Number of Missiles Required to Match Chicxulub Crater: {missiles_required:.0f}")

# Plot comparison
plt.figure(figsize=(10, 6))
objects = ['Missile Crater', 'Chicxulub Crater']
radii = [radius, asteroid_crater_radius]
plt.bar(objects, [r / 1000 for r in radii], color=['brown', 'gray'])
plt.title("Crater Radius Comparison")
plt.ylabel("Radius (km)")
plt.grid(True)
plt.show()
# Compare with comets of similar mass and different materials
comet_mass = missile_mass  # kg (same as missile mass)
comet_velocity = 50e3  # m/s (typical comet velocity)

# Materials: Earth (rock), lighter material (e.g., soil), and lead
materials = {
    "Earth (Rock)": {"density": target_density, "yield_strength": target_yield_strength},
    "Lighter Material": {"density": lighter_target_density, "yield_strength": lighter_target_yield_strength},
    "Lead": {"density": 11340, "yield_strength": 1.6e7}  # kg/m³ and Pa
}

# Calculate crater radius for each material
crater_radii = {}
for material, properties in materials.items():
    kinetic_energy = 0.5 * comet_mass * comet_velocity**2
    crater_volume = kinetic_energy / properties["yield_strength"]
    radius = (3 * crater_volume / (4 * np.pi))**(1/3)
    crater_radii[material] = radius

# Display results for each material
for material, radius in crater_radii.items():
    print(f"Material: {material}, Crater Radius: {radius:.2f} meters")

# Plot comparison of crater radii for different materials
plt.figure(figsize=(10, 6))
plt.bar(crater_radii.keys(), crater_radii.values(), color=['brown', 'tan', 'gray'])
plt.title("Crater Radius for Comet Impact on Different Materials")
plt.ylabel("Crater Radius (m)")
plt.grid(True)
plt.show()
# Compare missile and comet impact radii in different materials
objects = ['Missile (Rock)', 'Missile (Lighter Material)', 'Missile (Lead)',
           'Comet (Rock)', 'Comet (Lighter Material)', 'Comet (Lead)']
radii = [
    crater_radii["Earth (Rock)"], crater_radii["Lighter Material"], crater_radii["Lead"],
    crater_radii["Earth (Rock)"] * (comet_velocity / velocity)**(2/3),
    crater_radii["Lighter Material"] * (comet_velocity / velocity)**(2/3),
    crater_radii["Lead"] * (comet_velocity / velocity)**(2/3)
]

# Plot comparison
plt.figure(figsize=(12, 6))
plt.bar(objects, radii, color=['brown', 'tan', 'gray', 'brown', 'tan', 'gray'])
plt.title("Missile vs Comet Impact Radii in Different Materials")
plt.ylabel("Crater Radius (m)")
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()