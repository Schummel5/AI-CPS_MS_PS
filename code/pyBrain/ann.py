# This code solves the subgoal 4: AI Model PyBrain/Tensorflow Creation and Data Visualization
# These are the imports for this part of the task.
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
import scipy.stats as stats  
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from pybrain.structure import FullConnection

from pybrain.tools.shortcuts import buildNetwork

import pylab

# Checks if the path is available
output_dir = "/tmp/AIBAS_KURS_PS_MS/data/ANN/"
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

# print(train_dataset)
# print(y_test)
# print(y_train)
# print(test_dataset)


# Create ANN model
train = BackpropTrainer(n,train_dataset, learningrate=0.01, momentum=0.9)


# The following part trains the Ann model
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

# The following part saves the Ann model
NetworkWriter.writeToFile(n, os.path.join(output_dir, "currentAiSolution.xml"))
print(f"Model saved to XML file.")
#print(f"Train Errors: {train_errors[:5]}")
#print(f"Test Errors: {test_errors[:5]}")


# Create the trainings curve
plt.figure()
plt.plot(train_errors, label='Training Error', color='blue') # Train-Errors of each epoch
plt.plot(test_errors, label='Test Error', color='red') # Test-Errors of each epoch
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.title('Training & Test Error Curve')
plt.savefig(os.path.join(output_dir, "training_curve.png"))
print('Training Curve')

# Create the Error Plot 
predictions = np.array([n.activate(x)[0] for x in x_test])
errors = y_test - predictions
plt.figure()
plt.hist(errors, bins=25, alpha=0.7, color='blue')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Count')
plt.title('Error Distribution')
plt.savefig(os.path.join(output_dir, "error_distribution.png"))
print('ErrorPlot')

# Create the Residual Plot
plt.figure(figsize=(8, 6))
plt.scatter(predictions, errors, alpha=0.5, color="blue")
plt.axhline(y=0, color='r', linestyle='dashed')  
plt.xlabel("Predicted EPS")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.savefig(os.path.join(output_dir, "residual_plot.png"))
print('ResidualPlot')

# Create the Scatter Plot "Training vs. Testing vs. Predicitions"
plt.figure()
plt.scatter(x_train, y_train, color='orange', label='Training Data', alpha=0.7)
plt.scatter(x_test, y_test, color='blue', label='Testing Data', alpha=0.3)
plt.scatter(x_test, predictions, color='red', label='Predictions', alpha=0.5)
plt.xlabel('Estimated EPS')
plt.ylabel('Actual/Predicted EPS')
plt.title('Scatter Plot: Actual EPS vs. Predicted EPS')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='dashed', color='black')
plt.savefig(os.path.join(output_dir, "scatter_plot.png"))
print('ScatterPlot Training vs. Testing vs. Predicitions')

# Create the Scatter Plot "Actual vs. Predicted"
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5, color="purple")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="black", linestyle="dashed")
plt.xlabel("Actual EPS")
plt.ylabel("Predicted EPS")
plt.title("Actual vs. Predicted EPS")
plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"))
print('ScatterPlot Actual vs. Predicted')

# Create the Quantil-Quantil-Plot
stats.probplot(errors, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.savefig(os.path.join(output_dir, "qq_plot.png"))
print('Quantil-Quantil-Plot')

# Create the training_report.txt
with open(os.path.join(output_dir, "training_report.txt"), 'w') as f:
    f.write(f"Final Training Error: {train_errors[-1]:.4f}\n")
    f.write(f"Final Test Error: {test_errors[-1]:.4f}\n")

# Saving the error values of each epoch in a CSV file
train_test_loss_path = "/tmp/AIBAS_KURS_PS_MS/data/ANN/training_validation_loss.csv"
with open(train_test_loss_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Training Loss", "Test Loss"])
    for i in range(epochs):
        writer.writerow([i+1, train_errors[i], test_errors[i]])

# Saving the data of each training iteration
final_results_path = "/tmp/AIBAS_KURS_PS_MS/data/ANN/final_training_results.txt"
with open(final_results_path, "w") as file:
    file.write(f"Anzahl der Trainingsiterationen: {epochs}\n")
    file.write(f"Finaler Trainings-Loss: {train_errors[-1]:.4f}\n")
    file.write(f"Finaler Test-Loss: {test_errors[-1]:.4f}\n")

print(f"Finale Trainingsergebnisse gespeichert unter: {final_results_path}")
