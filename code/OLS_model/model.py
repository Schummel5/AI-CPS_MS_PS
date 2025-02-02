# import pandas and statsmodels

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import csv

output_dir = "/tmp/AIBAS_KURS_PS_MS/data/OLS_model/"
os.makedirs(output_dir, exist_ok=True)

# Reading the CSV files of the training and testing data
train_df = pd.read_csv('/tmp/AIBAS_KURS_PS_MS/data/training_data.csv')
test_df = pd.read_csv('/tmp/AIBAS_KURS_PS_MS/data/testing_data.csv')


print(train_df.head())

# OLS Model influence 'Estimated EPS', target 'Actual EPS'
x_train = train_df['Actual EPS'] # Independent variable
y_train = train_df['Estimated EPS'] # Dependent variable
x_test = test_df['Actual EPS']
y_test = test_df['Estimated EPS']

# add constant to predictor variables
X_train = sm.add_constant(x_train)
X_test = sm.add_constant(x_test)

# fit linear regression model
model = sm.OLS(y_train,X_train).fit()

'''print(len(x_train.index))
print(len(y_train.index))'''
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
print(f"Predictions shape: {model.predict(X_test).shape}")

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


#
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='orange', label='Training Data')
plt.scatter(x_test, y_test, color='blue', label='Testing Data', alpha=0.3)
plt.scatter(y_test, model.predict(X_test), color='red', alpha=0.5)
plt.xlabel("True Values")  # Updated label
plt.ylabel("Predictions")  # Updated label
plt.title("Scatter Plot OLS: True vs. Predicted")  # Updated title
plt.savefig(os.path.join(output_dir, "scatterplot_ols.png"))
plt.show()

residuals = y_test - model.predict(X_test)
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Residuals")  # Updated label
plt.ylabel("Count")  # Updated label
plt.title("Histogram of Residuals (Prediction Errors)")  # Updated title
plt.savefig(os.path.join(output_dir, "residualplot_ols.png"))
plt.show()

#view model summary
ols_model_file_path = '/tmp/AIBAS_KURS_PS_MS/data/OLS_model/currentOlsSolution.xml'
with open(ols_model_file_path, 'w') as file:
	file.write(model.summary().as_text())

