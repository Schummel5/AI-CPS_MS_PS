import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import sys 
sys.path.append('/tmp/learn/pybrain')

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from pybrain.tools.customxml import NetworkWriter, NetworkReader
import pylab

# Reading the CSV files of the training and testing data
train_df = pd.read_csv('/tmp/AIBAS_KURS_PS_MS/data/training_data')
test_df = pd.read_csv('/tmp/AIBAS_KURS_PS_MS/data/testing_data')


n = buildNetwork(1, 10, 10, 1, hiddenclass=SigmoidLayer, bias=True)

# PyBrain-Datasets für Training und Testing
train_dataset = SupervisedDataSet(1, 1)
test_dataset = SupervisedDataSet(1, 1)

for xi, yi in zip(x_train, y_train):
    train_dataset.addSample(xi, yi)

for xi, yi in zip(x_test, y_test):
    test_dataset.addSample(xi, yi)

print(train_dataset)
print(test_dataset)


# Create ANN Model
train = BackpropTrainer(n,train_dataset)


# Train the model
print("Training the model...")

train.trainEpochs(epochs=10)

scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

train_data_x = x_train.reshape(-1, 1)
train_data_y = y_train.reshape(-1, 1)


train_data_x_scaled = scaler_x.fit_transform(train_data_x)
train_data_y_scaled = scaler_y.fit_transform(train_data_y)


train_data_xx = train_data_x_scaled.ravel()
train_data_yy = train_data_y_scaled.ravel()


predictions_scaled = [n.activate([x])[0] for x in train_data_xx]
predictions = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

print("Training completed.")

# Save the model
model_file = "currentAiSoluGon.xml"
NetworkWriter.writeToFile(n, model_file)
print(f"Model saved to {model_file}.")


# Load the model
print("Loading the model...")
loaded_network = NetworkReader.readFrom(model_file)
print("Model loaded successfully.")

# Scatter Plot
# Create the plot
plt.figure(figsize=(10, 6))

# Scatter plots
plt.scatter(train_data['x'], train_data['y'], color='orange', label='Training Data', alpha=0.7)
plt.scatter(test_data['x'], test_data['y'], color='blue', label='Testing Data', alpha=0.3)


train_predictions = [n.activate([xi])[0] for xi in train_data['x'].values]

sorted_indices = np.argsort(train_data['x'].values)  # Indizes sortieren
sorted_x = train_data['x'].values[sorted_indices]   # Sortierte x-Werte
sorted_predictions = np.array(train_predictions)[sorted_indices]

plt.plot(sorted_x, predictions, color='red', label='Prediction Line (Trained Model)', linewidth=2)

# Add labels, title, legend
plt.xlabel('Influence Variable')
plt.ylabel('Target Variable')
plt.title('Scatter Plot with Training, Testing Data and Regression Line')
plt.legend()
plt.grid(alpha=0.3)

# Save the figure to a PDF
output_file = 'UE_06_ScatterVisualizationAndOlsModel.pdf'
plt.savefig(output_file, format='pdf')

# Show the plot
plt.show()

print(f"Figure saved as {output_file}.")

pylab.plot(sorted_x, train_predictions, color='blue', label='Prediction Line (Trained)', linewidth=2)
pylab.plot(train_data['x'], train_data['y'], color='red', label='Prediction Line (Trained)', linewidth=2)

pylab.grid()
pylab.legend()
pylab.show()