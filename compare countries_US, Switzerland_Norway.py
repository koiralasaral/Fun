import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- 1. Define Data for Each Country ---
countries = {
    'Turkmenistan': {
        'exports': {
            'Petroleum Gas': 10.9, 'Refined Petroleum': 2.73, 'Nitrogenous Fertilizers': 0.488,
            'Crude Petroleum': 0.378, 'Potassic Fertilizers': 0.208, 'Cotton': 0.1363,
            'Inorganic Chemicals': 0.056, 'Vegetables': 0.0439, 'Soaps and Lubricants': 0.0176, 'Others': 0.0429
        },
        'gdp_per_capita': 9191, 'trade_surplus': 10.8, 'population': 6.52, 'free_cash_flow': 2
    },
    'Uzbekistan': {
        'exports': {
            'Cotton': 2.3, 'Gold': 0.5, 'Natural Gas': 1.5, 'Fruits': 0.2, 'Machinery': 0.1, 'Others': 0.4
        },
        'gdp_per_capita': 1800, 'trade_surplus': 8, 'population': 34, 'free_cash_flow': 3
    },
    'Iran': {
        'exports': {
            'Petroleum Products': 60, 'Refined Petroleum': 20, 'Organic Chemicals': 5,
            'Automobiles': 2.5, 'Agricultural Products': 3, 'Others': 4
        },
        'gdp_per_capita': 5500, 'trade_surplus': 25, 'population': 85, 'free_cash_flow': 15
    },
    'Nepal': {
        'exports': {
            'Textiles': 0.5, 'Handicrafts': 0.3, 'Tea': 0.2, 'Herbs': 0.1, 'Others': 0.3
        },
        'gdp_per_capita': 1200, 'trade_surplus': -1, 'population': 30, 'free_cash_flow': -0.2
    },
    'United States': {
        'exports': {
            'Technology': 200, 'Automobiles': 150, 'Agriculture': 100, 'Pharmaceuticals': 75, 'Others': 50
        },
        'gdp_per_capita': 65000, 'trade_surplus': -50, 'population': 331, 'free_cash_flow': 500
    },
    'Switzerland': {
        'exports': {
            'Pharmaceuticals': 80, 'Machinery': 40, 'Metals': 30, 'Agriculture': 20, 'Others': 10
        },
        'gdp_per_capita': 88000, 'trade_surplus': 35, 'population': 8.6, 'free_cash_flow': 10
    },
    'Norway': {
        'exports': {
            'Oil': 90, 'Natural Gas': 60, 'Fish': 10, 'Technology': 5, 'Others': 5
        },
        'gdp_per_capita': 75000, 'trade_surplus': 40, 'population': 5.4, 'free_cash_flow': 15
    },
    'Estonia': {
        'exports': {
            'Technology': 5, 'Wood Products': 2, 'Machinery': 1.5, 'Agriculture': 1, 'Others': 0.5
        },
        'gdp_per_capita': 25000, 'trade_surplus': 2, 'population': 1.3, 'free_cash_flow': 0.5
    },
    'Latvia': {
        'exports': {
            'Wood Products': 4, 'Agriculture': 2.5, 'Machinery': 1, 'Technology': 0.8, 'Others': 0.7
        },
        'gdp_per_capita': 18000, 'trade_surplus': 1.5, 'population': 1.9, 'free_cash_flow': 0.3
    }
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
        'Free Cash Flow (Billion USD)': info['free_cash_flow']
    })

df_summary = pd.DataFrame(summary_data)
print("\nSummary Data:")
print(df_summary)

# --- 3. Comparative Export Concentration Analysis (Herfindahl-Hirschman Index - HHI) ---
def hhi(export_data):
    shares = np.array(list(export_data.values())) / sum(export_data.values())
    return np.sum(shares ** 2)

df_summary['Export Concentration (HHI)'] = df_summary['Country'].apply(lambda x: hhi(countries[x]['exports']))

# --- 4. Visualizations ---
plt.figure(figsize=(12, 6))
plt.bar(df_summary['Country'], df_summary['Export Concentration (HHI)'], color='purple')
plt.title("Export Concentration (HHI)")
plt.ylabel("HHI Index")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(df_summary['Total GDP (Billion USD)'], df_summary['Trade Surplus (Billion USD)'], color='blue')
plt.xlabel("Total GDP (Billion USD)")
plt.ylabel("Trade Surplus (Billion USD)")
plt.title("Trade Surplus vs. GDP")
plt.show()