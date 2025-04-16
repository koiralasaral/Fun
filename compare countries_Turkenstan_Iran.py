import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- 1. Define Data for Each Country ---
# For each country we store a dictionary:
# - 'exports': export values by category (in billion USD)
# - 'gdp_per_capita': in USD
# - 'trade_surplus': assumed net profit (billion USD; negative value indicates a net loss)
# - 'population': in millions
# - 'free_cash_flow': in billion USD
countries = {
'Turkmenistan': {
'exports': {
'Petroleum Gas': 10.9,
'Refined Petroleum': 2.73,
'Nitrogenous Fertilizers': 0.488,
'Crude Petroleum': 0.378,
'Potassic Fertilizers': 0.208,
'Cotton': 0.1363,
'Inorganic Chemicals': 0.056,
'Vegetables': 0.0439,
'Soaps and Lubricants': 0.0176,
'Others': 0.0429
},
'gdp_per_capita': 9191,
'trade_surplus': 10.8, # billion USD (net profit)
'population': 6.52, # million
'free_cash_flow': 2 # billion USD
},
'Uzbekistan': {
'exports': {
'Cotton': 2.3,
'Gold': 0.5,
'Natural Gas': 1.5,
'Fruits': 0.2,
'Machinery': 0.1,
'Others': 0.4
},
'gdp_per_capita': 1800,
'trade_surplus': 8, # billion USD
'population': 34, # million
'free_cash_flow': 3 # billion USD
},
'Iran': {
'exports': {
'Petroleum Products': 60,
'Refined Petroleum': 20,
'Organic Chemicals': 5,
'Automobiles': 2.5,
'Agricultural Products': 3,
'Others': 4
},
'gdp_per_capita': 5500,
'trade_surplus': 25, # billion USD
'population': 85, # million
'free_cash_flow': 15 # billion USD
},
'Afghanistan': {
'exports': {
'Carpets': 0.15,
'Fruits': 0.05,
'Precious Minerals': 0.02,
'Handicrafts': 0.03,
'Others': 0.05
},
'gdp_per_capita': 500,
'trade_surplus': -2, # billion USD (net loss)
'population': 38, # million
'free_cash_flow': -0.5 # billion USD
},
'Kazakhstan': {
'exports': {
'Oil': 30,
'Natural Gas': 10,
'Metals': 5,
'Grain': 2,
'Others': 3
},
'gdp_per_capita': 9500,
'trade_surplus': 15, # billion USD
'population': 19, # million
'free_cash_flow': 4 # billion USD
}
}

# --- 2. Create a Summary DataFrame for Comparative Metrics ---
summary_data = []
for country, info in countries.items():
	total_exports = sum(info['exports'].values())
	trade_surplus = info['trade_surplus'] # billion USD, acting as net profit (or loss)
	# Calculate total GDP (billion USD) = (GDP per capita * population) / 1000
	total_gdp = (info['gdp_per_capita'] * info['population']) / 1000
	# Profit per capita (in thousand USD): (trade surplus (in billion USD)*1000) / population
	profit_per_capita = (trade_surplus * 1000) / info['population']
	summary_data.append({
		'Country': country,
		'Total Exports (Billion USD)': total_exports,
		'GDP per Capita (USD)': info['gdp_per_capita'],
		'Trade Surplus (Billion USD)': trade_surplus,
		'Total GDP (Billion USD)': total_gdp,
		'Profit per Capita (Thousand USD)': profit_per_capita,
		'Free Cash Flow (Billion USD)': info['free_cash_flow']
	})

df_summary = pd.DataFrame(summary_data)
print("Summary Data:")
print(df_summary)

# --- 3. Analysis for Each Country ---
# For each country, perform regression analysis over export categories,
# compute statistical moments (mean, variance, skewness, kurtosis),
# and approximate the Moment Generating Function (MGF) at t=1.
for country, info in countries.items():
	print(f"\nAnalysis for {country}:")
	export_data = info['exports']
	df_exports = pd.DataFrame(list(export_data.items()), columns=['Category', 'Value (Billion USD)'])
	df_exports['Category Index'] = range(1, len(df_exports) + 1)

	# Calculate total exports and percentage share of each category
	total_exports = df_exports['Value (Billion USD)'].sum()
	df_exports['Percentage'] = (df_exports['Value (Billion USD)'] / total_exports) * 100

	print("Export Data:")
	print(df_exports)

	# Linear Regression: Export Value vs. Category Index
	slope, intercept, r_value, p_value, std_err = stats.linregress(df_exports['Category Index'], df_exports['Value (Billion USD)'])
	print("Regression Analysis:")
	print(f" Slope: {slope:.4f}")
	print(f" Intercept: {intercept:.4f}")
	print(f" R-squared: {r_value**2:.4f}")
	print(f" P-value: {p_value:.4f}")
	print(f" Standard Error: {std_err:.4f}")

	# Statistical Moments on Export Values
	mean_val = np.mean(df_exports['Value (Billion USD)'])
	variance_val = np.var(df_exports['Value (Billion USD)'])
	skewness_val = stats.skew(df_exports['Value (Billion USD)'])
	kurtosis_val = stats.kurtosis(df_exports['Value (Billion USD)'])

	print("Statistical Moments:")
	print(f" Mean: {mean_val:.4f}")
	print(f" Variance: {variance_val:.4f}")
	print(f" Skewness: {skewness_val:.4f}")
	print(f" Kurtosis: {kurtosis_val:.4f}")

	# Moment Generating Function (MGF) approximation at t=1
	t = 1
	mgf = np.mean(np.exp(t * df_exports['Value (Billion USD)']))
	print(f"Moment Generating Function (MGF) at t={t}: {mgf:.4f}")

	# Visualize Export Categories for the Country
	plt.figure(figsize=(8, 5))
	plt.bar(df_exports['Category'], df_exports['Value (Billion USD)'], color='skyblue')
	plt.xticks(rotation=45, ha='right')
	plt.title(f"{country}'s Exports by Category")
	plt.ylabel("Value (Billion USD)")
	plt.tight_layout()
	plt.show()

# --- 4. Comparative Visualizations for Summary Metrics ---

# Visualization: GDP per Capita, Total GDP, Trade Surplus, and Profit per Capita
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.bar(df_summary['Country'], df_summary['GDP per Capita (USD)'], color='skyblue')
plt.title("GDP per Capita")
plt.ylabel("USD")

plt.subplot(2, 2, 2)
plt.bar(df_summary['Country'], df_summary['Total GDP (Billion USD)'], color='lightgreen')
plt.title("Total GDP (Billion USD)")

plt.subplot(2, 2, 3)
plt.bar(df_summary['Country'], df_summary['Trade Surplus (Billion USD)'], color='salmon')
plt.title("Trade Surplus (Billion USD)")

plt.subplot(2, 2, 4)
plt.bar(df_summary['Country'], df_summary['Profit per Capita (Thousand USD)'], color='violet')
plt.title("Profit per Capita (Thousand USD)")

plt.tight_layout()
plt.show()

# Visualization: Free Cash Flow
plt.figure(figsize=(6, 4))
plt.bar(df_summary['Country'], df_summary['Free Cash Flow (Billion USD)'], color='gold')
plt.title("Free Cash Flow (Billion USD)")
plt.ylabel("Billion USD")
plt.tight_layout()
plt.show()