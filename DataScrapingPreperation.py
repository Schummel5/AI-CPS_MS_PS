import pylab
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
import csv
import requests
from bs4 import BeautifulSoup


# This file is for Data Scraping and the Preperation
# We scraped the Data from this URL: https://www.boerse-frankfurt.de/etf/amundi-msci-world-v-ucits-etf-acc/kurshistorie/historische-kurse-und-umsaetze?currency=EUR
# The target is to predict what the next Highs and Lows of this ETF is.

# Step 1:  Data Scraping

URL = "https://www.boerse-frankfurt.de/etf/amundi-msci-world-v-ucits-etf-acc/kurshistorie/historische-kurse-und-umsaetze?currency=EUR"
page = requests.get(URL).text

soup = BeautifulSoup(page, "html.parser")
table = soup.find("table")
rows = table.find_all("tr")
header = [th.text.strip() for th in rows[0].find_all("th")]

data = []
for row in rows[1:]:
    cells = row.find_all("td")
    data.append([cell.text.strip() for cell in cells])

df = pd.DataFrame(data, columns=header)


csv_file = "ScrapedDate.csv"
df.to_csv(csv_file, index=False, encoding="utf-8")

print(f"The data was successfully stored in the file: {csv_file}.")


# Step 2: Data Cleaning
