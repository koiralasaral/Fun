import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

# --- 1. Define Data for Each EU Country ---
# Updated dataset including next 5 countries (Netherlands, Belgium, Sweden, Austria, Poland)
countries = {
    'Germany': {'exports': {'Machinery': 200, 'Automobiles': 150, 'Chemicals': 100, 'Others': 50}, 'gdp_per_capita': 50000, 'trade_surplus': 250, 'population': 83, 'inflation': 3.2, 'debt_to_gdp': 65, 'fdi': 35, 'tax_rate': 25},
    'France': {'exports': {'Aerospace': 100, 'Automobiles': 80, 'Agriculture': 70, 'Others': 40}, 'gdp_per_capita': 45000, 'trade_surplus': 150, 'population': 67, 'inflation': 2.8, 'debt_to_gdp': 98, 'fdi': 30, 'tax_rate': 28},
    'Italy': {'exports': {'Luxury Goods': 90, 'Machinery': 75, 'Food': 60, 'Others': 35}, 'gdp_per_capita': 40000, 'trade_surplus': 100, 'population': 60, 'inflation': 3.0, 'debt_to_gdp': 140, 'fdi': 20, 'tax_rate': 24},
    'Spain': {'exports': {'Tourism': 80, 'Automobiles': 60, 'Food': 50, 'Others': 30}, 'gdp_per_capita': 38000, 'trade_surplus': 90, 'population': 47, 'inflation': 3.5, 'debt_to_gdp': 120, 'fdi': 18, 'tax_rate': 27},
    'Netherlands': {'exports': {'Technology': 150, 'Oil': 80, 'Agriculture': 70, 'Others': 40}, 'gdp_per_capita': 52000, 'trade_surplus': 200, 'population': 17, 'inflation': 2.5, 'debt_to_gdp': 65, 'fdi': 40, 'tax_rate': 26},
    'Belgium': {'exports': {'Chemicals': 90, 'Pharmaceuticals': 70, 'Machinery': 60, 'Others': 30}, 'gdp_per_capita': 48000, 'trade_surplus': 180, 'population': 11, 'inflation': 3.0, 'debt_to_gdp': 98, 'fdi': 25, 'tax_rate': 27},
    'Sweden': {'exports': {'Automobiles': 100, 'Technology': 80, 'Forestry': 50, 'Others': 35}, 'gdp_per_capita': 55000, 'trade_surplus': 220, 'population': 10, 'inflation': 2.2, 'debt_to_gdp': 40, 'fdi': 38, 'tax_rate': 22},
    'Austria': {'exports': {'Machinery': 70, 'Agriculture': 40, 'Luxury Goods': 35, 'Others': 25}, 'gdp_per_capita': 47000, 'trade_surplus': 160, 'population': 9, 'inflation': 2.7, 'debt_to_gdp': 75, 'fdi': 28, 'tax_rate': 23},
    'Poland': {'exports': {'Automobiles': 90, 'Technology': 50, 'Agriculture': 40, 'Others': 20}, 'gdp_per_capita': 38000, 'trade_surplus': 140, 'population': 38, 'inflation': 4.0, 'debt_to_gdp': 60, 'fdi': 22, 'tax_rate': 19},
}

# --- 2. Create Summary DataFrame ---
summary_data = []
for country, info in countries.items():
    total_exports = sum(info['exports'].values())
    total_gdp = (info['gdp_per_capita'] * info['population']) / 1000
    profit_per_capita = (info['trade_surplus'] * 1000) / info['population']
    
    summary_data.append({
        'Country': country,
        'Total Exports (Billion USD)': total_exports,
        'GDP per Capita (USD)': info['gdp_per_capita'],
        'Trade Surplus (Billion USD)': info['trade_surplus'],
        'Total GDP (Billion USD)': total_gdp,
        'Profit per Capita (Thousand USD)': profit_per_capita,
        'Inflation Rate (%)': info['inflation'],
        'Debt-to-GDP Ratio (%)': info['debt_to_gdp'],
        'FDI Inflows (Billion USD)': info['fdi'],
        'Corporate Tax Rate (%)': info['tax_rate']
    })

df_summary = pd.DataFrame(summary_data)
print("\nSummary Data:")
print(df_summary)

# --- 3. Correlation Matrix for Export Categories ---
export_matrix = pd.DataFrame({country: list(info['exports'].values()) for country, info in countries.items()})
correlation_matrix = export_matrix.corr()
print("\nExport Correlation Matrix:")
print(correlation_matrix)

# --- 4. Growth Forecast Using ARIMA ---
def forecast_exports(country_name, export_history):
    model = ARIMA(export_history, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=5)
    return forecast

# Simulated export history (adjust with real data)
historical_exports = {'Germany': [200, 210, 220, 230, 240], 'France': [100, 105, 110, 115, 120]}
for country in historical_exports:
    forecast = forecast_exports(country, historical_exports[country])
    print(f"\nExport Forecast for {country}: {forecast}")

# --- 5. Debt-to-GDP vs. Trade Surplus Visualization ---
plt.figure(figsize=(10, 6))
plt.scatter(df_summary['Debt-to-GDP Ratio (%)'], df_summary['Trade Surplus (Billion USD)'], color='red')
plt.xlabel("Debt-to-GDP Ratio (%)")
plt.ylabel("Trade Surplus (Billion USD)")
plt.title("Debt-to-GDP vs. Trade Surplus")
plt.show()