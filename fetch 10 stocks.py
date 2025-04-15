
import os
import time
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver

# --- Selenium setup ---
def setup_driver():
    service = Service("C:\\WebDriver\\chromedriver.exe")  # Update path if needed
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Keep the browser hidden
    driver = webdriver.Chrome(service=service, options=options)
    return driver

# --- Scraping functions ---
def get_all_symbols(driver, url):
    driver.get(url)
    symbols = []
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'table')))
        rows = driver.find_element(By.CLASS_NAME, 'table').find_elements(By.TAG_NAME, 'tr')[1:]
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, 'td')
            if cols:
                symbols.append(cols[0].text.strip())
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch symbols: {e}")
    return symbols

def get_company_details(driver, symbol):
    url = f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}"
    driver.get(url)
    details = {'Symbol': symbol}
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'table')))
        rows = driver.find_element(By.CLASS_NAME, 'table').find_elements(By.TAG_NAME, 'tr')
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, 'td')
            if len(cols) == 2:
                details[cols[0].text.strip()] = cols[1].text.strip()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch details for {symbol}: {e}")
    return details

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

# --- GUI ---
def run_scraper():
    driver = setup_driver()
    symbols = get_all_symbols(driver, "https://merolagani.com/LatestMarket.aspx")
    if not symbols:
        driver.quit()
        return

    results = []
    for i, symbol in enumerate(symbols[:10]):  # Scrape only 10 companies
        progress_label.config(text=f"Processing {i+1}/10: {symbol}")
        root.update()
        details = get_company_details(driver, symbol)
        results.append(details)
        time.sleep(1)

    driver.quit()

    save_to_csv(results, "company_details.csv")
    os.startfile("company_details.csv")  # Open file for viewing
    progress_label.config(text="Done! Data saved to company_details.csv")

# --- Tkinter GUI setup ---
root = tk.Tk()
root.title("MeroLagani Scraper")
root.geometry("400x200")

label = tk.Label(root, text="Scrape Top 10 Company Details from MeroLagani")
label.pack(pady=10)

run_button = tk.Button(root, text="Start Scraping", command=run_scraper)
run_button.pack(pady=5)

progress_label = tk.Label(root, text="")
progress_label.pack(pady=10)

root.mainloop()
