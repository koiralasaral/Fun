import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define howitzer specifications for scoring
howitzers = {
    'CAESAR NG': {'range': 60, 'rate': 8, 'auto': 'semi-auto', 'crew': 3, 'fire_control': True, 'autoloader': 'partial'},
    'Archer FH-77BW': {'range': 60, 'rate': 9, 'auto': 'full-auto', 'crew': 3, 'fire_control': True, 'autoloader': 'full'},
    'ATMOS 2020': {'range': 56, 'rate': 7, 'auto': 'semi-auto', 'crew': 4, 'fire_control': True, 'autoloader': 'partial'},
    'K9A2': {'range': 50, 'rate': 6, 'auto': 'full-auto', 'crew': 3, 'fire_control': True, 'autoloader': 'full'},
    'M1299': {'range': 70, 'rate': 10, 'auto': 'full-auto', 'crew': 3, 'fire_control': True, 'autoloader': 'full'},
    'PLZ-52A': {'range': 53, 'rate': 8, 'auto': 'full-auto', 'crew': 3, 'fire_control': True, 'autoloader': 'full'}
}

# Define weight multipliers for scores
max_range = 70  # Highest range observed
max_rate = 10   # Highest fire rate observed
crew_factor = {2: 1.0, 3: 0.9, 4: 0.8}  # Crew impact factor
auto_level_factor = {'semi-auto': 0.9, 'full-auto': 1.0}  # Automation scoring
fire_control_factor = {True: 1.0, False: 0.8}  # Fire control efficiency
autoloader_factor = {'none': 0.8, 'partial': 0.9, 'full': 1.0}  # Autoloader effect

# Compute effectiveness scores
howitzer_scores = {}
for name, specs in howitzers.items():
    score = (
        (specs['range'] / max_range) + 
        (specs['rate'] / max_rate) + 
        auto_level_factor[specs['auto']] +
        crew_factor[specs['crew']] + 
        fire_control_factor[specs['fire_control']] + 
        autoloader_factor[specs['autoloader']]
    ) / 6  # Normalize across attributes
    howitzer_scores[name] = round(score, 2)

# Compute energy based on shell velocity (Joules)
shell_mass = 45  # kg
muzzle_velocity = {'CAESAR NG': 850, 'Archer FH-77BW': 870, 'ATMOS 2020': 830, 'K9A2': 860, 'M1299': 890, 'PLZ-52A': 845}

howitzer_energy = {}
for name, velocity in muzzle_velocity.items():
    energy = 0.5 * shell_mass * (velocity ** 2)  # E = 1/2 m v^2
    howitzer_energy[name] = round(energy, -6)  # Rounded for visualization

# Convert data to DataFrame
df_scores = pd.DataFrame(list(howitzer_scores.items()), columns=["Howitzer", "Effectiveness Score"])
df_energy = pd.DataFrame(list(howitzer_energy.items()), columns=["Howitzer", "Destruction Energy (Joules)"])

# Print Results
print("\nHowitzer Effectiveness Scores:")
print(df_scores.to_markdown(index=False))

print("\nHowitzer Energy Values (Joules):")
print(df_energy.to_markdown(index=False))

# Visualization
plt.figure(figsize=(12, 6))

# Plot Effectiveness Scores
plt.subplot(1, 2, 1)
sns.barplot(x="Howitzer", y="Effectiveness Score", data=df_scores, palette="coolwarm")
plt.title("Howitzer Effectiveness Comparison")
plt.xticks(rotation=45)

# Plot Destruction Energy
plt.subplot(1, 2, 2)
sns.barplot(x="Howitzer", y="Destruction Energy (Joules)", data=df_energy, palette="magma")
plt.title("Howitzer Destruction Capability")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# --- Define Howitzer Effectiveness Scores ---
howitzer_scores = {
    'CAESAR NG': 0.85,
    'Archer FH-77BW': 0.92,
    'ATMOS 2020': 0.78,
    'K9A2': 0.88,
    'M1299': 0.95,
    'PLZ-52A': 0.82
}

# --- Ownership Data ---
owners_data = {
    'Country': ['France', 'Sweden', 'Israel', 'South Korea', 'USA', 'China', 'Poland', 'Egypt', 'India'],
    'CAESAR NG': [24, 0, 0, 0, 0, 0, 0, 12, 0],
    'Archer FH-77BW': [0, 24, 0, 0, 0, 0, 48, 0, 0],
    'ATMOS 2020': [0, 0, 36, 0, 0, 0, 0, 24, 0],
    'K9A2': [0, 0, 0, 72, 0, 0, 48, 0, 100],
    'M1299': [0, 0, 0, 0, 18, 0, 0, 0, 0],
    'PLZ-52A': [0, 0, 0, 0, 0, 150, 0, 0, 0]
}

owners_df = pd.DataFrame(owners_data)

# Function to extract howitzers per country
def get_howitzers_for_country(country):
    row = owners_df[owners_df["Country"] == country]
    howitzers = []
    for col in owners_df.columns[1:]:
        howitzers.extend([col] * row[col].values[0])
    return howitzers if howitzers else ["None"]

# --- Monte Carlo Simulation ---
battle_results = {country: 0 for country in owners_df["Country"]}
num_simulations = 1000
np.random.seed(42)

def simulate_battle(blue_howitzer, red_howitzer):
    blue_score = howitzer_scores[blue_howitzer] + random.uniform(-0.05, 0.05)
    red_score = howitzer_scores[red_howitzer] + random.uniform(-0.05, 0.05)
    return blue_score > red_score

for _ in range(num_simulations):
    blue_country, red_country = random.sample(list(owners_df["Country"]), 2)
    blue_howitzer = random.choice(get_howitzers_for_country(blue_country))
    red_howitzer = random.choice(get_howitzers_for_country(red_country))

    if blue_howitzer != "None" and red_howitzer != "None":
        if simulate_battle(blue_howitzer, red_howitzer):
            battle_results[blue_country] += 1
        else:
            battle_results[red_country] += 1

# --- Visualization of Battle Results ---
plt.figure(figsize=(12, 6))
sns.barplot(x=list(battle_results.keys()), y=list(battle_results.values()), palette="husl")
plt.title("Monte Carlo Simulation: Battle Victories Over 1000 Engagements (Using Ownership Data)")
plt.ylabel("Number of Wins")
plt.xlabel("Country")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# --- Manufacturer Trends Visualization ---
mfg_trends = {
    'Year': [2020, 2021, 2022, 2023, 2024],
    'Nexter_CAESAR_NG': [10, 15, 20, 25, 30],
    'BAE_Archer_BW': [8, 12, 18, 24, 30],
    'Elbit_ATMOS': [15, 20, 25, 30, 35],
    'Hanwha_K9A2': [30, 45, 60, 75, 90],
    'BAE_M1299': [0, 2, 5, 10, 15],
    'NORINCO_PLZ52A': [40, 60, 80, 100, 120]
}

mfg_df = pd.DataFrame(mfg_trends)
mfg_df.plot(x='Year', title='Modern Howitzer Production Trends (2020-2024)')
plt.ylabel('Units Produced Annually')
plt.xlabel('Year')
plt.grid(True)
plt.show()

# --- Destruction Radii Calculations ---
material_density = {'rock': 2700, 'earth': 1500, 'lead': 11340} 
material_strength = {'rock': 1e7, 'earth': 5e6, 'lead': 2e7}  

def calculate_destruction(energy, material):
    strength = material_strength[material]
    radius = (3 * energy / (4 * np.pi * strength)) ** (1/3)  
    return radius

howitzer_energy = {
    'CAESAR NG': 1e9,
    'Archer FH-77BW': 1.2e9,
    'ATMOS 2020': 0.9e9,
    'K9A2': 1.1e9,
    'M1299': 1.5e9,
    'PLZ-52A': 1.3e9
}

destruction_data = []
for howitzer, energy in howitzer_energy.items():
    for material in material_density.keys():
        radius = calculate_destruction(energy, material)
        destruction_data.append({'Howitzer': howitzer, 'Material': material, 'Radius (m)': radius})

destruction_df = pd.DataFrame(destruction_data)

plt.figure(figsize=(12, 6))
sns.barplot(x='Material', y='Radius (m)', hue='Howitzer', data=destruction_df, palette='coolwarm')
plt.title("Destruction Radius Comparison")
plt.ylabel("Impact Radius (m)")
plt.grid(True)
plt.show()

print("\nFinal Battle Results Over 1000 Simulations:")
for country, wins in sorted(battle_results.items(), key=lambda x: x[1], reverse=True):
    print(f"{country}: {wins} wins")