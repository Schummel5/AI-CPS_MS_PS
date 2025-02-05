import statsmodels.api as sm
import pandas as pd
import pickle


with open('/tmp/ai_system/knowledgeBase/currentOlsSolution.pkl','rb') as f: #/tmp/AIBAS_KURS_PS_MS/images/knowledgeBase/currentOlsSolution.pkl
  model = pickle.load(f)


act_df = pd.read_csv(f'/tmp/ai_system/activationBase/activation_data.csv')#/tmp/AIBAS_KURS_PS_MS/data/activation_data.csv

x_train = act_df['Estimated EPS'] # Independent variable
y_train = act_df['Actual EPS'] # Dependent variable

#X_train = sm.add_constant(x_train)


prediction = model.predict(act_df)

print("The predicted EPS is:", prediction)
print("The actual EPS is:", act_df['Actual EPS'])
