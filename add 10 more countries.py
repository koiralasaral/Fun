import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

# --- 1. Define Data for Each EU Country ---
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

# --- 3. Correlation Analysis Graph ---
export_matrix = pd.DataFrame({country: list(info['exports'].values()) for country, info in countries.items()})
correlation_matrix = export_matrix.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Export Categories Correlation Across EU Countries")
plt.show()

# --- 4. ARIMA Forecasting Visualization ---
def forecast_exports(country_name, export_history):
    model = ARIMA(export_history, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=5)
    
    # Plot historical vs forecasted values
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(export_history)), export_history, marker="o", label="Historical Exports")
    plt.plot(range(len(export_history), len(export_history) + len(forecast)), forecast, marker="x", linestyle="dashed", label="Forecast")
    plt.title(f"Export Growth Forecast - {country_name}")
    plt.legend()
    plt.show()
    
    return forecast

# Example export history (adjust with real data)
historical_exports = {'Germany': [200, 210, 220, 230, 240], 'France': [100, 105, 110, 115, 120]}
for country in historical_exports:
    forecast_exports(country, historical_exports[country])

# --- 5. Financial Metrics Visualization ---
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(df_summary['Country'], df_summary['Debt-to-GDP Ratio (%)'], color='red')
plt.title("Debt-to-GDP Ratio Across EU Countries")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar(df_summary['Country'], df_summary['FDI Inflows (Billion USD)'], color='green')
plt.title("FDI Inflows Comparison")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
# --- 6. Export Types and Popular Products Analysis ---

# Add export types and most bought product for each country
export_details = {
    'Germany': {'companies': {'Siemens': 'Machinery', 'Volkswagen': 'Automobiles', 'BASF': 'Chemicals'}, 'most_bought': 'Volkswagen Automobiles'},
    'France': {'companies': {'Airbus': 'Aerospace', 'Peugeot': 'Automobiles', 'Danone': 'Agriculture'}, 'most_bought': 'Airbus Aerospace'},
    'Italy': {'companies': {'Gucci': 'Luxury Goods', 'Fiat': 'Automobiles', 'Barilla': 'Food'}, 'most_bought': 'Gucci Luxury Goods'},
    'Spain': {'companies': {'Iberia': 'Tourism', 'SEAT': 'Automobiles', 'Gallo': 'Food'}, 'most_bought': 'Iberia Tourism'},
    'Netherlands': {'companies': {'ASML': 'Technology', 'Shell': 'Oil', 'Heineken': 'Agriculture'}, 'most_bought': 'ASML Technology'},
    'Belgium': {'companies': {'Solvay': 'Chemicals', 'UCB': 'Pharmaceuticals', 'Atlas Copco': 'Machinery'}, 'most_bought': 'Solvay Chemicals'},
    'Sweden': {'companies': {'Volvo': 'Automobiles', 'Ericsson': 'Technology', 'SCA': 'Forestry'}, 'most_bought': 'Volvo Automobiles'},
    'Austria': {'companies': {'Red Bull': 'Luxury Goods', 'KTM': 'Machinery', 'Agrana': 'Agriculture'}, 'most_bought': 'Red Bull Luxury Goods'},
    'Poland': {'companies': {'Solaris': 'Automobiles', 'CD Projekt': 'Technology', 'Maspex': 'Agriculture'}, 'most_bought': 'Solaris Automobiles'},
}

# Categorize exports by company name and type
export_categories = []
for country, details in export_details.items():
    for company, product_type in details['companies'].items():
        export_categories.append({
            'Country': country,
            'Company': company,
            'Product Type': product_type,
            'Most Bought Product': details['most_bought']
        })

df_exports = pd.DataFrame(export_categories)

# Display categorized exports
print("Categorized Exports by Company and Type:")
print(df_exports)

# Visualization of export types
plt.figure(figsize=(12, 6))
sns.countplot(data=df_exports, y='Product Type', order=df_exports['Product Type'].value_counts().index, palette="viridis")
plt.title("Export Types Distribution Across EU Countries")
plt.xlabel("Count")
plt.ylabel("Product Type")
plt.show()