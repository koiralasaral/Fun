import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Removed unused import: tabulate
# Removed unused import: Axes3D
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [14, 8]

# Specifications data for modern howitzers (post-2020)
modern_specs = {
    'Specification': ['Caliber', 'Max Range', 'Rate of Fire', 'Automation Level', 
                     'Crew Size', 'Mobility', 'Weight', 'Unit Cost', 'First Deployment',
                     'Digital Fire Control', 'Autoloader'],
    'CAESAR NG (France 2021)': ['155mm', '60 km', '8 rpm', 'Semi-auto', 
                                '2-3', '8x8 truck', '22 t', '$6.2M', '2023',
                                'Yes', 'Partial'],
    'Archer FH-77BW (Sweden 2022)': ['155mm', '60 km', '9 rpm', 'Fully auto', 
                                    '2-3', '6x6 truck', '32 t', '$5.8M', '2022',
                                    'Yes', 'Full'],
    'ATMOS 2020 (Israel)': ['155mm', '56 km', '7 rpm', 'Semi-auto', 
                           '3-4', '6x6 truck', '24 t', '$5.1M', '2021',
                           'Yes', 'Partial'],
    'K9A2 (South Korea 2021)': ['155mm', '50 km', '6 rpm', 'Fully auto', 
                               '3', 'Tracked', '47 t', '$4.9M', '2022',
                               'Yes', 'Full'],
    'M1299 (USA 2023)': ['155mm', '70 km', '10 rpm', 'Fully auto', 
                        '3', 'Tracked', '45 t', '$7.5M', '2024',
                        'Yes', 'Full'],
    'PLZ-52A (China 2022)': ['155mm', '53 km', '8 rpm', 'Fully auto', 
                            '3', 'Tracked', '42 t', '$4.2M', '2023',
                            'Yes', 'Full']
}

modern_df = pd.DataFrame(modern_specs)
print("Modern Howitzer Specifications Comparison:")
print(modern_df.to_markdown(index=False))

# Ownership data for modern systems
modern_owners = {
    'Country': ['France', 'Sweden', 'Israel', 'South Korea', 
               'USA', 'China', 'Poland', 'Egypt', 'India'],
    'CAESAR NG': [24, 0, 0, 0, 0, 0, 0, 12, 0],
    'Archer FH-77BW': [0, 24, 0, 0, 0, 0, 48, 0, 0],
    'ATMOS 2020': [0, 0, 36, 0, 0, 0, 0, 24, 0],
    'K9A2': [0, 0, 0, 72, 0, 0, 48, 0, 100],
    'M1299': [0, 0, 0, 0, 18, 0, 0, 0, 0],
    'PLZ-52A': [0, 0, 0, 0, 0, 150, 0, 0, 0]
}
# Ownership data for modern systems
owners_df = pd.DataFrame(modern_owners)
numeric_columns = owners_df.select_dtypes(include=[np.number]).columns
owners_df['Total Modern'] = owners_df[numeric_columns].sum(axis=1)

# Monte Carlo simulation for each country based on specifications
np.random.seed(42)
num_battles = 1000

# Assign scores to specifications for simulation
spec_scores = {
    'CAESAR NG (France 2021)': {
        'Max Range': 3,
        'Rate of Fire': 2,
        'Automation Level': 1,
        'Crew Size': -1,
        'Mobility': 1,
        'Weight': -1,
        'Unit Cost': -2,
        'Digital Fire Control': 2,
        'Autoloader': 1
    },
    'Archer FH-77BW (Sweden 2022)': {
        'Max Range': 3,
        'Rate of Fire': 3,
        'Automation Level': 2,
        'Crew Size': -1,
        'Mobility': 1,
        'Weight': -2,
        'Unit Cost': -2,
        'Digital Fire Control': 2,
        'Autoloader': 2
    },
    'ATMOS 2020 (Israel)': {
        'Max Range': 2,
        'Rate of Fire': 2,
        'Automation Level': 1,
        'Crew Size': -1,
        'Mobility': 1,
        'Weight': -1,
        'Unit Cost': -1,
        'Digital Fire Control': 2,
        'Autoloader': 1
    },
    'K9A2 (South Korea 2021)': {
        'Max Range': 2,
        'Rate of Fire': 2,
        'Automation Level': 2,
        'Crew Size': -1,
        'Mobility': 1,
        'Weight': -2,
        'Unit Cost': -1,
        'Digital Fire Control': 2,
        'Autoloader': 2
    },
    'M1299 (USA 2023)': {
        'Max Range': 4,
        'Rate of Fire': 3,
        'Automation Level': 2,
        'Crew Size': -1,
        'Mobility': 1,
        'Weight': -2,
        'Unit Cost': -3,
        'Digital Fire Control': 2,
        'Autoloader': 2
    },
    'PLZ-52A (China 2022)': {
        'Max Range': 3,
        'Rate of Fire': 2,
        'Automation Level': 2,
        'Crew Size': -1,
        'Mobility': 1,
        'Weight': -2,
        'Unit Cost': -1,
        'Digital Fire Control': 2,
        'Autoloader': 2
    }
}

# Prepare data for simulation
country_scores = {}
for _, row in owners_df.iterrows():
    total_score = 0
    for howitzer, count in row.items():
        if howitzer != 'Country' and howitzer != 'Total Modern' and count > 0:
            howitzer_specs = modern_df[modern_df['Specification'].isin(spec_scores[howitzer].keys())][howitzer].values
            for spec, value in zip(modern_df['Specification'], howitzer_specs):
                if spec in spec_scores[howitzer]:
                    # Convert numeric values where applicable
                    if isinstance(value, str):
                        if ' km' in value or ' rpm' in value or ' t' in value:
                            value = float(value.split()[0])
                        elif value.startswith('$'):
                            value = float(value[1:].replace('M', '')) * 1e6  # Convert cost to numeric
                        elif value.isdigit():
                            value = int(value)
                        elif '-' in value:  # Handle ranges like '2-3'
                            value = sum(map(float, value.split('-'))) / 2  # Take the average
                    total_score += spec_scores[howitzer][spec] * value * count
    country_scores[row['Country']] = total_score

# Simulate battles
battle_results = {country: 0 for country in country_scores.keys()}
for _ in range(num_battles):
    countries = list(country_scores.keys())
    np.random.shuffle(countries)
    country1, country2 = countries[:2]
    score1, score2 = country_scores[country1], country_scores[country2]
    if score1 + np.random.randint(-10, 10) > score2 + np.random.randint(-10, 10):
        battle_results[country1] += 1
    else:
        battle_results[country2] += 1

# Determine the country with the most wins
most_wins_country = max(battle_results, key=battle_results.get)
print(f"\nCountry with the most wins: {most_wins_country} ({battle_results[most_wins_country]} wins)")

# Display battle results
battle_results_df = pd.DataFrame(
    [{'Country': country, 'Wins': wins} for country, wins in battle_results.items()]
).sort_values(by='Wins', ascending=False)

print("\nBattle Results by Country:")
print(battle_results_df.to_markdown(index=False))

# Plot battle results
battle_results_df.plot(x='Country', y='Wins', kind='bar', color='green', legend=False)
plt.title('Monte Carlo Simulation: Battle Wins by Country')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
owners_df = pd.DataFrame(modern_owners)
numeric_columns = owners_df.select_dtypes(include=[np.number]).columns
owners_df['Total Modern'] = owners_df[numeric_columns].sum(axis=1)

# Plot ownership
owners_df.plot(x='Country', kind='bar', stacked=True, 
               title='Modern Howitzer Ownership by Country (Post-2020 Systems)')
plt.ylabel('Number of Units')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Manufacturer trends
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

# Plot production trends
mfg_df.plot(x='Year', title='Modern Howitzer Production Trends (2020-2024)')
plt.ylabel('Units Produced Annually')
plt.xlabel('Year')
plt.grid(True)
plt.show()

# Similarity analysis
similar_features = modern_df[
    (modern_df['CAESAR NG (France 2021)'] == modern_df['Archer FH-77BW (Sweden 2022)']) |
    (modern_df['Specification'].isin(['Caliber', 'Automation Level', 'Digital Fire Control']))
]

print("\nCommon Features Among Modern Howitzers:")
print(similar_features.to_markdown(index=False))

# Range comparison
range_df = modern_df[modern_df['Specification'] == 'Max Range'].T[1:]
range_df.columns = ['Max Range']
range_df['Max Range'] = range_df['Max Range'].str.replace(' km', '').astype(float)
range_df = range_df.sort_values('Max Range', ascending=False)

range_df.plot(kind='bar', legend=False, color='orange')
plt.title('Maximum Range Comparison (km)')
plt.ylabel('Kilometers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Prepare data for 3D plot
features = ['Max Range', 'Rate of Fire', 'Weight']
data = modern_df[modern_df['Specification'].isin(features)].T[1:]
data.columns = features

# Convert units and normalize data
data['Max Range'] = data['Max Range'].str.replace(' km', '').astype(float)
data['Rate of Fire'] = data['Rate of Fire'].str.replace(' rpm', '').astype(float)
data['Weight'] = data['Weight'].str.replace(' t', '').astype(float)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Encode labels for howitzers
howitzers = modern_df.columns[1:]
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(howitzers)

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(scaled_data[:, 0], scaled_data[:, 1], scaled_data[:, 2], c=encoded_labels, cmap='viridis', s=100)
ax.set_title('3D Analysis of Modern Howitzers')
ax.set_xlabel('Max Range (Normalized)')
ax.set_ylabel('Rate of Fire (Normalized)')
ax.set_zlabel('Weight (Normalized)')

# Add annotations
for i, label in enumerate(howitzers):
    ax.text(scaled_data[i, 0], scaled_data[i, 1], scaled_data[i, 2], label, fontsize=8)

plt.tight_layout()
plt.show()
# Constants for material properties (in J/m^3)
material_density = {'rock': 2700, 'earth': 1500, 'lead': 11340}  # kg/m^3
material_strength = {'rock': 1e7, 'earth': 5e6, 'lead': 2e7}  # Pa (N/m^2)

# Function to calculate energy and radius of destruction
def calculate_destruction(energy, material):
    # Removed unused variable 'density'
    strength = material_strength[material]
    radius = (3 * energy / (4 * np.pi * strength)) ** (1/3)  # Simplified model
    return radius

# Example energy values (in Joules) for each howitzer
howitzer_energy = {
    'CAESAR NG': 1e9,
    'Archer FH-77BW': 1.2e9,
    'ATMOS 2020': 0.9e9,
    'K9A2': 1.1e9,
    'M1299': 1.5e9,
    'PLZ-52A': 1.3e9
}

# Calculate destruction radii for each material and howitzer
destruction_data = []
for howitzer, energy in howitzer_energy.items():
    for material in material_density.keys():
        radius = calculate_destruction(energy, material)
        destruction_data.append({'Howitzer': howitzer, 'Material': material, 'Radius (m)': radius})

destruction_df = pd.DataFrame(destruction_data)
print("\nDestruction Radii (meters):")
print(destruction_df.to_markdown(index=False))
# Probability of winning 1000 battles with different combinations of howitzers
np.random.seed(42)
num_battles = 1000
blue_force = 50
red_force = 50

win_probabilities = {}

for blue_howitzer in howitzer_energy.keys():
    for red_howitzer in howitzer_energy.keys():
        blue_wins = 0
        for _ in range(num_battles):
            blue_remaining = blue_force
            red_remaining = red_force
            while blue_remaining > 0 and red_remaining > 0:
                blue_remaining -= np.random.randint(1, 5)  # Random attrition
                red_remaining -= np.random.randint(1, 5)
            if blue_remaining > red_remaining:
                blue_wins += 1
        win_probabilities[(blue_howitzer, red_howitzer)] = blue_wins / num_battles

# Display win probabilities
win_prob_df = pd.DataFrame(
    [
        {'Blue Howitzer': blue, 'Red Howitzer': red, 'Win Probability': prob}
        for (blue, red), prob in win_probabilities.items()
    ]
)

print("\nWin Probabilities for Different Howitzer Combinations:")
print(win_prob_df.to_markdown(index=False))

# Heatmap of win probabilities
win_prob_pivot = win_prob_df.pivot(index='Blue Howitzer', columns='Red Howitzer', values='Win Probability')
sns.heatmap(win_prob_pivot, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'Win Probability'})
plt.title('Win Probability Heatmap (Blue vs Red Howitzers)')
plt.xlabel('Red Howitzer')
plt.ylabel('Blue Howitzer')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
# Monte Carlo simulation to compare the number of CAESAR, ATMOS, K9A2, M1299, and PLZ-52A needed to win
np.random.seed(42)
required_comparisons = []

selected_howitzers = ['CAESAR NG', 'ATMOS 2020', 'K9A2', 'M1299', 'PLZ-52A']

for blue_howitzer in selected_howitzers:
    for red_howitzer in selected_howitzers:
        for num_blue_howitzers in range(1, 101):  # Test with 1 to 100 blue howitzers
            for num_red_howitzers in range(1, 101):  # Test with 1 to 100 red howitzers
                blue_wins = 0
                for _ in range(num_battles):
                    blue_remaining = num_blue_howitzers * blue_force
                    red_remaining = num_red_howitzers * red_force
                    while blue_remaining > 0 and red_remaining > 0:
                        blue_remaining -= np.random.randint(1, 5)  # Random attrition
                        red_remaining -= np.random.randint(1, 5)
                    if blue_remaining > red_remaining:
                        blue_wins += 1
                win_probability = blue_wins / num_battles
                if win_probability >= 0.75:  # Define 75% win probability as the threshold
                    required_comparisons.append({
                        'Blue Howitzer': blue_howitzer,
                        'Red Howitzer': red_howitzer if 'red_howitzer' in locals() else None,
                        'Blue Howitzers Needed': num_blue_howitzers,
                        'Red Howitzers Needed': num_red_howitzers
                    })
                    break
            if win_probability >= 0.75:
                break

# Display required comparisons
required_comparisons_df = pd.DataFrame(required_comparisons)

print("\nComparison of Howitzer Numbers to Achieve 75% Win Probability:")
print(required_comparisons_df.to_markdown(index=False))

# Heatmap for required blue howitzers against red howitzers
pivot_blue = required_comparisons_df.pivot(index='Blue Howitzer', columns='Red Howitzer', values='Blue Howitzers Needed')
sns.heatmap(pivot_blue, annot=True, cmap='Blues', fmt=".0f", cbar_kws={'label': 'Blue Howitzers Needed'})
plt.title('Blue Howitzers Needed for 75% Win Probability (Selected Howitzers)')
plt.xlabel('Red Howitzer')
plt.ylabel('Blue Howitzer')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Heatmap for required red howitzers against blue howitzers
pivot_red = required_comparisons_df.pivot(index='Blue Howitzer', columns='Red Howitzer', values='Red Howitzers Needed')
sns.heatmap(pivot_red, annot=True, cmap='Reds', fmt=".0f", cbar_kws={'label': 'Red Howitzers Needed'})
plt.title('Red Howitzers Needed for 75% Win Probability (Selected Howitzers)')
plt.xlabel('Red Howitzer')
plt.ylabel('Blue Howitzer')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
num_battles = 1000
blue_force = 50
red_force = 50
battle_results = {}

for howitzer in howitzer_energy.keys():
    blue_losses = []
    red_losses = []
    for _ in range(num_battles):
        blue_remaining = blue_force
        red_remaining = red_force
        while blue_remaining > 0 and red_remaining > 0:
            blue_remaining -= np.random.randint(1, 5)  # Random attrition
            red_remaining -= np.random.randint(1, 5)
        blue_losses.append(blue_force - max(blue_remaining, 0))
        red_losses.append(red_force - max(red_remaining, 0))
    battle_results[howitzer] = {'Blue Losses': blue_losses, 'Red Losses': red_losses}

# Visualize battle outcomes for each howitzer
for howitzer, results in battle_results.items():
    plt.hist(results['Blue Losses'], bins=20, alpha=0.7, label='Blue Losses', color='blue')
    plt.hist(results['Red Losses'], bins=20, alpha=0.7, label='Red Losses', color='red')
    plt.title(f'Monte Carlo Battle Outcomes ({howitzer})')
    plt.xlabel('Losses')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Lanchester's attrition model for each howitzer
time_steps = 100
blue_force = 50
red_force = 50
blue_power = 0.8
red_power = 0.9

for howitzer in howitzer_energy.keys():
    blue_history = [blue_force]
    red_history = [red_force]
    blue_force_temp = blue_force
    red_force_temp = red_force

    for _ in range(time_steps):
        blue_force_temp -= red_power * red_force_temp / 100
        red_force_temp -= blue_power * blue_force_temp / 100
        blue_history.append(max(blue_force_temp, 0))
        red_history.append(max(red_force_temp, 0))
        if blue_force_temp <= 0 or red_force_temp <= 0:
            break

    # Plot Lanchester's attrition for the current howitzer
    plt.plot(blue_history, label='Blue Force', color='blue')
    plt.plot(red_history, label='Red Force', color='red')
    plt.title(f"Lanchester's Attrition Model ({howitzer})")
    plt.xlabel('Time Steps')
    plt.ylabel('Force Strength')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Prepare data for 3D plot of destruction radii
    materials = list(material_density.keys())
    howitzers = list(howitzer_energy.keys())
    radii = []

    for howitzer in howitzers:
        radii_row = []
        for material in materials:
            radius_row = destruction_df[(destruction_df['Howitzer'] == howitzer) & (destruction_df['Material'] == material)]
            if not radius_row.empty:
                radius = radius_row['Radius (m)'].values[0]
            else:
                radius = 0  # Default value if no match is found
            radii_row.append(radius)
        radii.append(radii_row)

    radii = np.array(radii)

  
    # Create subplots for each material
    fig = plt.figure(figsize=(16, 12))
    for i, material in enumerate(materials):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')

        # Extract radii for the current material
        z = radii[:, i]

        # Create a meshgrid for plotting the crater
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        # Calculate the crater shape based on the radius
        for j, radius in enumerate(z):
            distance = np.sqrt(X**2 + Y**2)
            crater_depth = np.maximum(0, radius - distance * radius)
            Z -= crater_depth / len(z)  # Normalize depth for visualization

        # Plot the crater
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

        # Add labels and title
        ax.set_title(f'Crater Visualization for {material}')
        ax.set_xlabel('X (Normalized)')
        ax.set_ylabel('Y (Normalized)')
        ax.set_zlabel('Depth (Normalized)')

    plt.tight_layout()
    plt.show()
    # Separate 3D Bar Plots for Different Materials in Subplots
    fig = plt.figure(figsize=(16, 12))

    for i, material in enumerate(materials):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')

        # Prepare data for the current material
        x_pos = np.arange(len(howitzers))
        y_pos = np.zeros_like(x_pos)
        z_pos = np.zeros_like(x_pos)
        bar_heights = radii[:, i]

        # Bar dimensions
        dx = dy = 0.5
        dz = bar_heights

        # Plot 3D bars for the current material
        ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, shade=True, color='skyblue')

        # Set axis labels and ticks
        ax.set_xlabel('Howitzers')
        ax.set_ylabel('Material')
        ax.set_zlabel('Radius (m)')
        ax.set_xticks(np.arange(len(howitzers)))
        ax.set_xticklabels(howitzers, rotation=45, ha='right')
        ax.set_yticks([0])
        ax.set_yticklabels([material])

        # Title for the subplot
        ax.set_title(f'Destruction Radii for {material}')

    plt.tight_layout()
    plt.show()
