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

## Step 1:  Data Scraping

URL = "https://www.boerse-frankfurt.de/etf/amundi-msci-world-v-ucits-etf-acc/kurshistorie/historische-kurse-und-umsaetze?currency=EUR"
page = requests.get(URL).content

soup = BeautifulSoup(page, "html.parser")

table = soup.find("table")

header = [header.text.strip() for header in table.find_all("th")]

rows = table.find_all("tr")

data = []
for row in rows[1:]:
    cells = row.find_all("td")
    data.append([cell.text.strip() for cell in cells])

df = pd.DataFrame(data, columns=header)


csv_file = "ScrapedDate.csv"
df.to_csv(csv_file, index=False)

print(f"The data was successfully stored in the file: {csv_file}.")


## Step 2: Data Cleaning

df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Quantile Filter
lower_quantile = 0.10
upper_quantile = 0.90

for column in df.select_dtypes(include='number'):
    lower_bound = df[column].quantile(lower_quantile)
    upper_bound = df[column].quantile(upper_quantile)
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


# Z-Score Filter
z_threshold = 3

for column in df.select_dtypes(include='number'):
	df['zscore'] = zscore(df[column])
	df = df[(df['zscore'].abs() <= z_threshold)]

df = df.drop(columns=['zscore'])

# IQR_Filter
for column in df.select_dtypes(include='number'):
	Q1 = df[column].quantile(0.25)
	Q3 = df[column].quantile(0.75)
	IQR = Q3 - Q1

	lower_bound = Q1 - 1.5 * IQR
	upper_bound = Q3 + 1.5 * IQR
	df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

