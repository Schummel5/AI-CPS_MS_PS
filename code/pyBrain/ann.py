import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import sys 
sys.path.append('/tmp/AIBAS_KURS_PS_MS/pybrain')
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer
from pybrain.structure.connections import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml import NetworkWriter, NetworkReader
from sklearn.preprocessing import StandardScaler



import statsmodels.api as sm

import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from pybrain.structure import FullConnection

from pybrain.tools.shortcuts import buildNetwork

import pylab


output_dir = "/tmp/AIBAS_KURS_PS_MS/data/"
os.makedirs(output_dir, exist_ok=True)

# Reading the CSV files of the training and testing data
train_df = pd.read_csv('/tmp/AIBAS_KURS_PS_MS/data/training_data.csv')
test_df = pd.read_csv('/tmp/AIBAS_KURS_PS_MS/data/test_data.csv')

# Collecting the x and y of each dataset
x_train, y_train = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
x_test, y_test = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values


# Creating the ANN network
n = buildNetwork(1, 10, 10, 1, hiddenclass=SigmoidLayer, bias=True)

# Creating the PyBrain-Datasets for training und testing
train_dataset = SupervisedDataSet(x_train.shape[1], 1)
test_dataset = SupervisedDataSet(x_test.shape[1], 1)

for i in range(len(x_train)):
    train_dataset.addSample(x_train[i], y_train[i])

for i in range(len(x_test)):
    test_dataset.addSample(x_test[i], y_test[i])

print(train_dataset)
print(test_dataset)


# Create ANN Model
train = BackpropTrainer(n,train_dataset, learningrate=0.01, momentum=0.9)


# Train the model
print("Training the model...")

epochs = 50
train_errors = []
test_errors = []


for epoch in range(epochs):
    train_error = train.train()
    train_error = np.mean(train_error) 
    train_errors.append(train_error)
    test_error = train.testOnData(test_dataset)
    test_errors.append(test_error)
    
print(f"Train error type: {type(train_error)}")
print(f"Train error value: {train_error}")

print(f"Epoch {epoch+1}/{epochs}: Training Error = {train_error:.4f}, Test Error = {test_error:.4f}")

print("Training completed.")

# Save the model
NetworkWriter.writeToFile(n, os.path.join(output_dir, "currentAiSolution.xml"))
print(f"Model saved to XML file.")
#print(f"Train Errors: {train_errors[:5]}")
#print(f"Test Errors: {test_errors[:5]}")


# Create the trainings curve
plt.figure()
plt.plot(train_errors, label='Training Error', color='blue')
plt.plot(test_errors, label='Test Error', color='red')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.title('Training & Test Error Curve')
plt.savefig(os.path.join(output_dir, "training_curve.png"))
print('Training Curve')

# Create the error Plot
predictions = np.array([n.activate(x)[0] for x in x_test])
errors = y_test - predictions
plt.figure()
plt.hist(errors, bins=25, alpha=0.7, color='blue')
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title('Error Distribution')
plt.savefig(os.path.join(output_dir, "error_distribution.png"))
print('ErrorPlot')

# Scatter Plot
plt.figure()
plt.scatter(x_train, y_train, color='orange', label='Training Data', alpha=0.7)
plt.scatter(x_test, y_test, color='blue', label='Testing Data', alpha=0.3)
plt.scatter(y_test, predictions, color='red', label='Predictions', alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Scatter Plot: True vs. Predicted')
plt.savefig(os.path.join(output_dir, "scatter_plot.png"))
print('ScatterPlot')

with open(os.path.join(output_dir, "training_report.txt"), 'w') as f:
    f.write(f"Final Training Error: {train_errors[-1]:.4f}\n")
    f.write(f"Final Test Error: {test_errors[-1]:.4f}\n")