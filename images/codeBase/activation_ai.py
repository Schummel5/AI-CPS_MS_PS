import sys
sys.path.append('/tmp/codeBase/pybrain') #/tmp/AIBAS_KURS_PS_MS/pybrain
from pybrain.tools.customxml.networkreader import NetworkReader
import pandas as pd

ann = NetworkReader.readFrom(f'/tmp/ai_system/knowledgeBase/currentAiSolution.xml ') #/tmp/AIBAS_KURS_PS_MS/data/ANN/currentAiSolution.xml
act_df = pd.read_csv(f'/tmp/ai_system/activationBase/activation_data.csv') #/tmp/AIBAS_KURS_PS_MS/data/activation_data.csv

x_test = act_df[['Estimated EPS']]
#print(x_test)
#print(act_df.head())
print('The predicted value is:', ann.activate(x_test))
