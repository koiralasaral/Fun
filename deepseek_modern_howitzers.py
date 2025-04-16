import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

# Enhanced visualization settings
plt.style.use('ggplot')  # Replacing with a default style
plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['font.size'] = 12
colors = sns.color_palette("husl", 8)

# Extended specifications data with more technical parameters
modern_specs_ext = {
    'Specification': ['Caliber', 'Max Range', 'Rate of Fire', 'Automation Level', 
                     'Crew Size', 'Mobility', 'Weight', 'Unit Cost', 'First Deployment',
                     'Digital Fire Control', 'Autoloader', 'Time to Action', 
                     'Reload Time', 'Navigation System', 'Protection Level'],
    'CAESAR NG (France 2021)': ['155mm', '60 km', '8 rpm', 'Semi-auto', 
                               '2-3', '8x8 truck', '22 t', '$6.2M', '2023',
                               'Yes', 'Partial', '1 min', '2 min', 'Inertial+GPS', 'STANAG 4569 L2'],
    'Archer FH-77BW (Sweden 2022)': ['155mm', '60 km', '9 rpm', 'Fully auto', 
                                    '2-3', '6x6 truck', '32 t', '$5.8M', '2022',
                                    'Yes', 'Full', '30 sec', '1 min', 'Inertial+GNSS', 'STANAG 4569 L3'],
    'ATMOS 2020 (Israel)': ['155mm', '56 km', '7 rpm', 'Semi-auto', 
                           '3-4', '6x6 truck', '24 t', '$5.1M', '2021',
                           'Yes', 'Partial', '45 sec', '1.5 min', 'GPS', 'STANAG 4569 L2'],
    'K9A2 (South Korea 2021)': ['155mm', '50 km', '6 rpm', 'Fully auto', 
                               '3', 'Tracked', '47 t', '$4.9M', '2022',
                               'Yes', 'Full', '1 min', '2 min', 'Inertial+GPS', 'STANAG 4569 L4'],
    'M1299 (USA 2023)': ['155mm', '70 km', '10 rpm', 'Fully auto', 
                        '3', 'Tracked', '45 t', '$7.5M', '2024',
                        'Yes', 'Full', '40 sec', '45 sec', 'Inertial+GNSS', 'STANAG 4569 L4'],
    'PLZ-52A (China 2022)': ['155mm', '53 km', '8 rpm', 'Fully auto', 
                            '3', 'Tracked', '42 t', '$4.2M', '2023',
                            'Yes', 'Full', '50 sec', '1 min', 'BeiDou+INS', 'STANAG 4569 L3']
}

modern_df_ext = pd.DataFrame(modern_specs_ext)

# Enhanced ownership data with geopolitical context
modern_owners_ext = {
    'Country': ['France', 'Sweden', 'Israel', 'South Korea', 
               'USA', 'China', 'Poland', 'Egypt', 'India', 'Ukraine'],
    'Region': ['Europe', 'Europe', 'Middle East', 'Asia', 
              'Americas', 'Asia', 'Europe', 'Africa', 'Asia', 'Europe'],
    'CAESAR NG': [24, 0, 0, 0, 0, 0, 0, 12, 0, 18],
    'Archer FH-77BW': [0, 24, 0, 0, 0, 0, 48, 0, 0, 0],
    'ATMOS 2020': [0, 0, 36, 0, 0, 0, 0, 24, 0, 0],
    'K9A2': [0, 0, 0, 72, 0, 0, 48, 0, 100, 0],
    'M1299': [0, 0, 0, 0, 18, 0, 0, 0, 0, 0],
    'PLZ-52A': [0, 0, 0, 0, 0, 150, 0, 0, 0, 0]
}

owners_df_ext = pd.DataFrame(modern_owners_ext)
numeric_cols = owners_df_ext.select_dtypes(include=np.number).columns
owners_df_ext['Total'] = owners_df_ext[numeric_cols].sum(axis=1)

# Enhanced visualization 1: Radial chart for key specifications
specs_to_compare = ['Max Range', 'Rate of Fire', 'Weight', 'Unit Cost', 'Time to Action']
howitzers = modern_df_ext.columns[1:]
num_vars = len(specs_to_compare)

# Convert and normalize data
plot_data = []
for spec in specs_to_compare:
    row = modern_df_ext[modern_df_ext['Specification'] == spec].values[0][1:]
    cleaned = [float(x.split()[0]) if isinstance(x, str) and x[0].isdigit() else 
              float(x[1:].replace('M', '')) if isinstance(x, str) and x.startswith('$') else 
              0 for x in row]
    plot_data.append(cleaned)

plot_df = pd.DataFrame(plot_data, columns=howitzers, index=specs_to_compare)
scaler = MinMaxScaler()
plot_normalized = scaler.fit_transform(plot_df)

# Create radar chart
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Close the loop

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)

for idx, howitzer in enumerate(howitzers):
    values = plot_normalized[:, idx].tolist()
    values += values[:1]  # Close the loop
    ax.plot(angles, values, linewidth=2, linestyle='solid', 
            label=howitzer.split(' (')[0], color=colors[idx])
    ax.fill(angles, values, alpha=0.1, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(specs_to_compare)
ax.set_rlabel_position(30)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["20%", "40%", "60%", "80%"], color="grey", size=10)
plt.ylim(0, 1)
plt.title('Modern Howitzer Capability Radar Chart', pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

# Enhanced visualization 2: Geopolitical deployment map (simulated)
regions = owners_df_ext['Region'].unique()
region_totals = owners_df_ext.groupby('Region')[numeric_cols].sum()

fig, ax = plt.subplots(figsize=(14, 8))
bottom = np.zeros(len(regions))
for i, col in enumerate(numeric_cols):
    ax.bar(regions, region_totals[col], bottom=bottom, label=col.split(' (')[0], color=colors[i])
    bottom += region_totals[col]

ax.set_title('Global Deployment by Region')
ax.set_ylabel('Number of Units')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Enhanced visualization 3: 3D Cost-Effectiveness Analysis
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Prepare data
costs = [float(x[1:].replace('M', '')) for x in modern_df_ext[modern_df_ext['Specification'] == 'Unit Cost'].values[0][1:]]
ranges = [float(x.split()[0]) for x in modern_df_ext[modern_df_ext['Specification'] == 'Max Range'].values[0][1:]]
fire_rates = [float(x.split()[0]) for x in modern_df_ext[modern_df_ext['Specification'] == 'Rate of Fire'].values[0][1:]]

# Size of bubbles represents deployment numbers
sizes = owners_df_ext[numeric_cols].sum().values * 10

ax.scatter(costs, ranges, fire_rates, s=sizes, c=range(len(howitzers)), cmap='viridis', alpha=0.7)

# Annotations
for i, howitzer in enumerate(howitzers):
    ax.text(costs[i], ranges[i], fire_rates[i], howitzer.split(' (')[0], fontsize=10)

ax.set_xlabel('Unit Cost ($M)')
ax.set_ylabel('Max Range (km)')
ax.set_zlabel('Rate of Fire (rpm)')
ax.set_title('3D Cost-Effectiveness Analysis (Bubble Size = Global Deployment)')
plt.tight_layout()
plt.show()

# Enhanced tactical simulation with Lanchester's laws
def lanchester_square(blue_units, red_units, blue_effectiveness, red_effectiveness):
    return blue_units**2 - (red_effectiveness/blue_effectiveness) * red_units**2

# Assign effectiveness scores based on specifications
effectiveness = {
    'CAESAR NG (France 2021)': 0.85,
    'Archer FH-77BW (Sweden 2022)': 0.92,
    'ATMOS 2020 (Israel)': 0.78,
    'K9A2 (South Korea 2021)': 0.88,
    'M1299 (USA 2023)': 0.95,
    'PLZ-52A (China 2022)': 0.82
}

# Simulate engagements
results = []
for blue in howitzers:
    for red in howitzers:
        if blue != red:
            outcome = lanchester_square(50, 50, effectiveness[blue], effectiveness[red])
            results.append({
                'Blue Force': blue.split(' (')[0],
                'Red Force': red.split(' (')[0],
                'Outcome': 'Blue Wins' if outcome > 0 else 'Red Wins',
                'Advantage': abs(outcome)
            })

results_df = pd.DataFrame(results)

# Create advantage matrix
pivot_df = results_df.pivot(index='Blue Force', columns='Red Force', values='Advantage')

plt.figure(figsize=(12, 10))
sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="coolwarm", 
            cbar_kws={'label': 'Tactical Advantage Score'})
plt.title('Lanchester Model Engagement Outcomes\n(50 vs 50 units)')
plt.xlabel('Red Force Howitzer')
plt.ylabel('Blue Force Howitzer')
plt.tight_layout()
plt.show()

# Enhanced production timeline with cumulative view
mfg_trends_ext = {
    'Year': [2020, 2021, 2022, 2023, 2024, 2025],
    'Nexter (CAESAR NG)': [10, 25, 45, 70, 100, 135],
    'BAE (Archer BW)': [8, 20, 38, 62, 92, 130],
    'Elbit (ATMOS)': [15, 35, 60, 90, 125, 165],
    'Hanwha (K9A2)': [30, 75, 135, 210, 300, 405],
    'BAE (M1299)': [0, 2, 7, 17, 32, 52],
    'NORINCO (PLZ52A)': [40, 100, 180, 280, 400, 540]
}

mfg_df_ext = pd.DataFrame(mfg_trends_ext).set_index('Year')

plt.figure(figsize=(14, 8))
mfg_df_ext.plot(kind='line', marker='o')
plt.title('Cumulative Production Projections (2020-2025)')
plt.ylabel('Total Units Produced')
plt.xlabel('Year')
plt.grid(True)
plt.legend(title='Manufacturer')
plt.tight_layout()
plt.show()

# Add mobility vs protection analysis
mobility_types = {
    'CAESAR NG (France 2021)': 'Wheeled (8x8)',
    'Archer FH-77BW (Sweden 2022)': 'Wheeled (6x6)',
    'ATMOS 2020 (Israel)': 'Wheeled (6x6)',
    'K9A2 (South Korea 2021)': 'Tracked',
    'M1299 (USA 2023)': 'Tracked',
    'PLZ-52A (China 2022)': 'Tracked'
}

protection_levels = {
    'CAESAR NG (France 2021)': 2,
    'Archer FH-77BW (Sweden 2022)': 3,
    'ATMOS 2020 (Israel)': 2,
    'K9A2 (South Korea 2021)': 4,
    'M1299 (USA 2023)': 4,
    'PLZ-52A (China 2022)': 3
}

mobility_df = pd.DataFrame({
    'Howitzer': [x.split(' (')[0] for x in mobility_types.keys()],
    'Mobility Type': mobility_types.values(),
    'Protection Level': protection_levels.values()
})

plt.figure(figsize=(12, 6))
sns.scatterplot(data=mobility_df, x='Mobility Type', y='Protection Level', 
                hue='Howitzer', s=200, palette=colors)
plt.title('Mobility vs Protection Trade-off')
plt.yticks([1, 2, 3, 4], ['STANAG L1', 'STANAG L2', 'STANAG L3', 'STANAG L4'])
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()