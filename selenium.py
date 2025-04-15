
from selenium.webdriver.chrome.service import Service

service = Service("C:\\WebDriver\\chromedriver.exe")  # Replace with your actual ChromeDriver path
driver = webdriver.Chrome(service=service)
driver.get("https://www.google.com")
print("WebDriver is working!")
driver.quit()