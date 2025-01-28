# import pandas and statsmodels

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file

path = '/tmp/AIBAS_KURS_PS_MS/data/.csv'
data = pd.read_csv(path)

print(data.head())

# 6. OLS Model influence 'Estimated EPS', target 'Actual EPS'
# response variable endog = y
y = data['Actual EPS']
# explanatory variable exog = x
x = data['Estimated EPS']

# add constant to predictor variables
x = sm.add_constant(x)

# fit linear regression model
model = sm.OLS(y,x).fit()

#view model summary
ols_model_file_path = '/tmp/AIBAS_KURS_PS_MS/data/OLS_model/currentOlsSolution.xml'
with open(ols_model_file_path, 'w') as file:
	file.write(model.summary().as_text())
