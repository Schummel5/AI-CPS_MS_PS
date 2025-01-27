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


## Step 2: Data Cleaning
path_cleaned = '/tmp/AIBAS_KURS_PS_MS/data/cleaned_data.csv'

df.iloc[:, -2] = df.iloc[:, -2].replace({'\$': '', ',': ''}, regex=True).astype(float) 
df.iloc[:, -1] = df.iloc[:, -1].replace({'\$': '', ',': ''}, regex=True).astype(float)  
df_cleaned = df.iloc[:, 2:]
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(axis=1, how='any')
#print(df_cleaned.head())
df_cleaned.to_csv(folder_cleaned, index=False)

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

path_normalized_filtered_cleaned_data = '/tmp/AIBAS_KURS_PS_MS/data/normalized_filtered_cleaned_data.csv'

df.to_csv(path_normalized_filtered_cleaned_data, index=False)

## Step 3: Split the data in training- and testingdata
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

path_train = '/tmp/AIBAS_KURS_PS_MS/data/training_data.csv'
path_test= '/tmp/AIBAS_KURS_PS_MS/data/testing_data.csv'

train_data.to_csv(path_train, index=False)
test_data.to_csv(path_test, index=False)
