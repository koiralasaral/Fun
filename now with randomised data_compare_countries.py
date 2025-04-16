import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# List of 27 EU countries
eu_countries = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
    "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
    "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
    "Spain", "Sweden"
]

# --- 1. Generate Synthetic Data ---
np.random.seed(42)  # for reproducibility

# For demonstration, we assume:
# - Exports: values between 20 and 500 Billion USD, with larger economies (Germany, France, Italy) at the top end.
# - GDP per capita: values from 15,000 USD to 90,000 USD
# - Trade surplus: could be positive or negative (in Billion USD)
# - Population: between 0.5 million (Luxembourg, Malta) and 80 million (Germany)
# - Free cash flow: a proxy metric in Billion USD between 0 and 100

# We can design these using random values weighted by the expected economic size:
exports = []
gdp_per_capita = []
trade_surplus = []
population = []
free_cash_flow = []

for country in eu_countries:
    if country in ["Germany", "France", "Italy", "Netherlands"]:
        exp_val = np.random.uniform(200, 500)
        gdp_val = np.random.uniform(40000, 90000)
        pop_val = np.random.uniform(50, 80) if country == "Germany" else np.random.uniform(10, 50)
    elif country in ["Luxembourg", "Malta", "Cyprus"]:
        exp_val = np.random.uniform(20, 100)
        gdp_val = np.random.uniform(50000, 90000)
        pop_val = np.random.uniform(0.5, 2)
    else:
        exp_val = np.random.uniform(50, 200)
        gdp_val = np.random.uniform(15000, 40000)
        pop_val = np.random.uniform(3, 20)
        
    exports.append(round(exp_val, 2))
    gdp_per_capita.append(round(gdp_val, 0))
    trade_surplus.append(round(np.random.uniform(-30, 60), 2))  # surplus or deficit
    population.append(round(pop_val, 2))
    free_cash_flow.append(round(np.random.uniform(1, 100), 2))

# Create DataFrame
data = {
    "Country": eu_countries,
    "Exports (Billion USD)": exports,
    "GDP per Capita (USD)": gdp_per_capita,
    "Trade Surplus (Billion USD)": trade_surplus,
    "Population (Millions)": population,
    "Free Cash Flow (Billion USD)": free_cash_flow
}

df = pd.DataFrame(data)
df["Profit per Capita (Thousand USD)"] = (df["Trade Surplus (Billion USD)"] * 1000) / df["Population (Millions)"]

print("Synthetic EU Summary Data:")
print(df)

# --- 2. Correlation Analysis ---
# Compute Pearson correlations between variables
corr_matrix = df[["Exports (Billion USD)", "GDP per Capita (USD)",
                  "Trade Surplus (Billion USD)", "Population (Millions)",
                  "Free Cash Flow (Billion USD)", "Profit per Capita (Thousand USD)"]].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Visualize correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix for Synthetic EU Data")
plt.show()

# --- 3. Growth Forecasts ---
# For simplicity, we simulate synthetic historical export values over 10 years for each country.
# Assume past growth for each country: initial value from our "Exports" and an average annual growth rate sampled from N(2%, 1%).
years = np.arange(2015, 2025)
growth_rates = np.random.normal(loc=0.02, scale=0.01, size=len(eu_countries))  # synthetic annual growth rate for each country

forecast_data = {}
for i, country in enumerate(eu_countries):
    initial_value = df.loc[df["Country"] == country, "Exports (Billion USD)"].values[0]
    rate = growth_rates[i]
    exports_history = [initial_value * ((1 + rate) ** (year - 2015)) for year in years]
    forecast_data[country] = exports_history

# Convert to DataFrame
df_forecast = pd.DataFrame(forecast_data, index=years)
print("\nSynthetic Historical Exports (Billion USD) for Selected Years:")
print(df_forecast.head())

# Fit a simple linear regression (time vs exports) for a sample country and forecast next 5 years
sample_country = "Germany" if "Germany" in df["Country"].values else eu_countries[0]
y = df_forecast[sample_country].values
X = years.reshape(-1, 1)
X_sm = sm.add_constant(X)  # add constant term

model = sm.OLS(y, X_sm).fit()
print(f"\nGrowth Forecast Regression Summary for {sample_country}:")
print(model.summary())

# Forecast next 5 years
future_years = np.arange(2025, 2030)
X_future = sm.add_constant(future_years)
forecast_values = model.predict(X_future)
df_forecast_future = pd.DataFrame({
    "Year": future_years,
    f"{sample_country} Forecast Exports": forecast_values
})
print("\nForecasted Exports for Next 5 Years for", sample_country)
print(df_forecast_future)

plt.figure(figsize=(10, 6))
plt.plot(years, y, marker='o', label="Historical")
plt.plot(future_years, forecast_values, marker='o', linestyle='--', color='red', label="Forecast")
plt.xlabel("Year")
plt.ylabel("Exports (Billion USD)")
plt.title(f"Historical and Forecast Exports for {sample_country}")
plt.legend()
plt.show()

# --- 4. Further Financial Insights ---
# Here we highlight some additional synthetic analyses. For example, we analyze whether Free Cash Flow is positively associated with Export performance.
plt.figure(figsize=(8, 6))
sns.regplot(x="Free Cash Flow (Billion USD)", y="Exports (Billion USD)", data=df)
plt.title("Relationship between Free Cash Flow and Exports")
plt.xlabel("Free Cash Flow (Billion USD)")
plt.ylabel("Exports (Billion USD)")
plt.show()

# Additional insight: Assess if countries with higher GDP per capita tend to have higher exports.
plt.figure(figsize=(8, 6))
sns.scatterplot(x="GDP per Capita (USD)", y="Exports (Billion USD)", data=df, hue="Country", legend=False)
plt.title("GDP per Capita vs. Exports (Billion USD)")
plt.xlabel("GDP per Capita (USD)")
plt.ylabel("Exports (Billion USD)")
plt.show()

# We can also compute a simple export concentration metric, for example using a synthetic Herfindahl-Hirschman Index (HHI)
# Assuming each country exports a mix of 5 synthetic product categories with random shares for demonstration.
def compute_hhi(shares):
    shares = np.array(shares) / np.sum(shares)
    return np.sum(shares ** 2)

np.random.seed(2025)
hhi_values = []
for _ in eu_countries:
    # generate 5 random export category values
    categories = np.random.rand(5)
    hhi = compute_hhi(categories)
    hhi_values.append(round(hhi, 3))

df["Export Concentration (HHI)"] = hhi_values

print("\nEnhanced EU Data with Export Concentration (HHI):")
print(df[["Country", "Exports (Billion USD)", "Export Concentration (HHI)"]])

# You could extend this further by correlating HHI with trade surplus or free cash flow if desired.
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Export Concentration (HHI)", y="Trade Surplus (Billion USD)", data=df)
plt.title("Export Concentration (HHI) vs. Trade Surplus")
plt.xlabel("Export Concentration (HHI)")
plt.ylabel("Trade Surplus (Billion USD)")
plt.show()
