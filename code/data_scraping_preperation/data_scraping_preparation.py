# This file is for Data Scraping and Preparation and solves
# the subgoal 2: Data Scraping and Preparation
# We scraped the Data from this URL: https://www.alphaquery.com/stock/AAPL/earnings-history

# Here are the necessary imports for the data scraping
import csv
import requests
import os
from bs4 import BeautifulSoup
import pandas as pd

# Here are the necessary import for the data preperation
import pylab
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
import csv
import requests
import os
from scipy.stats import  zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

## Step 1:  Data Scraping by defining the URL
URL = "https://www.alphaquery.com/stock/AAPL/earnings-history"
page = requests.get(URL)
directory = "/tmp/AIBAS_KURS_PS_MS/data/"

# Checks if the directory data exists
os.makedirs(directory, exist_ok=True)

# This code checks if there is a table on the website (URL) and saves it in a CSV file
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

    csv_file_path = "/tmp/AIBAS_KURS_PS_MS/data/scraped_data.csv"
    df.to_csv(csv_file_path, index=False)

    print(f"The data was successfully stored in the file: {csv_file_path}.")
else:
    print("Keine Tabelle gefunden.")


## Step 2: Data Cleaning
path_cleaned = '/tmp/AIBAS_KURS_PS_MS/data/cleaned_data.csv'
# Change the data to numeric
df.iloc[:, -2] = df.iloc[:, -2].replace({r'\$': '', ',': ''}, regex=True).astype(float)
df.iloc[:, -1] = df.iloc[:, -1].replace({r'\$': '', ',': ''}, regex=True).astype(float)
df_cleaned = df.iloc[:, 2:]
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(axis=1, how='any')
#print(df_cleaned.head())
df_cleaned.to_csv(path_cleaned, index=False)

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

path_normalized_filtered_cleaned_data = '/tmp/AIBAS_KURS_PS_MS/data/joint_data_collection.csv'

df.to_csv(path_normalized_filtered_cleaned_data, index=False)

## Step 3: Split the data in training- and testingdata
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

path_train = '/tmp/AIBAS_KURS_PS_MS/data/training_data.csv'
path_test= '/tmp/AIBAS_KURS_PS_MS/data/test_data.csv'

train_data.to_csv(path_train, index=False)
test_data.to_csv(path_test, index=False)

## Step 4: Split the data in activation_data

activation_data = test_data.iloc[:1]

path_activation = '/tmp/AIBAS_KURS_PS_MS/data/activation_data.csv'

activation_data.to_csv(path_activation, index=False)




