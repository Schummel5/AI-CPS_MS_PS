import pylab
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
import csv
import requests
from bs4 import BeautifulSoup
from scipy.stats import  zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


# This file is for Data Scraping and the Preperation
# We scraped the Data from this URL: https://www.boerse-frankfurt.de/etf/amundi-msci-world-v-ucits-etf-acc/kurshistorie/historische-kurse-und-umsaetze?currency=EUR
# The target is to predict what the next Highs and Lows of this ETF is.

## Step 1:  Data Scraping
URL = "https://www.alphaquery.com/stock/AAPL/earnings-history"
#URL = "https://finance.yahoo.com/quote/AAPL/history/"
page = requests.get(URL)

#######Debugging########
#print(f"Statuscode: {page.status_code}")
#print(f"Headers: {page.headers}")
#print(f"Content-Type: {page.headers.get('Content-Type')}")
#print(f"Text: {page.text[:500]}")

soup = BeautifulSoup(page.text, "html.parser")

table = soup.find("table")

if table:
    rows = table.find_all("tr")
    data = []
    for row in rows:
        cells = row.find_all("td")
        if cells:
	        data.append([cell.text.strip() for cell in cells])

    header = [th.text.strip() for th in table.find_all("th")]

    df = pd.DataFrame(data, columns=header)

    csv_file = "ScrapedData.csv"
    df.to_csv(csv_file, index=False)

    print(f"The data was successfully stored in the file: {csv_file}.")
else:
    print("Keine Tabelle gefunden.")

#print(df)

## Step 2: Data Cleaning


df.iloc[:, -2] = df.iloc[:, -2].replace({'\$': '', ',': ''}, regex=True).astype(float) 
df.iloc[:, -1] = df.iloc[:, -1].replace({'\$': '', ',': ''}, regex=True).astype(float)  
df_cleaned = df.iloc[:, 2:]
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(axis=1, how='any')
#print(df_cleaned.head())
df_cleaned.to_csv("cleaned_data.csv", index=False)

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


# Data normalization with Min-Max Normalization
numerical_columns = df.select_dtypes(include='number').columns
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
#print(df)


df.to_csv("normalized_filtered_cleaned_data.csv", index=False)

## Step 3: Split the data in training- and testingdata
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

train_data.to_csv('Data_training.csv', index=False)
test_data.to_csv('Data_testing.csv', index=False)



