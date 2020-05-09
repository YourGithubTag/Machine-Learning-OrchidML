import matplotlib.pyplot as plt
import numpy as np
import csv

# This function plots the graphs for training, validation and testing.
# Interactive mode is disabled and all graphs are saved to the root directory with appropriate names.
def plot_graphs_csv(x, y, y_label, label, modelname):
    plt.ioff()
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    figfilename = label + "_" + modelname + ".png"
    csvfilename = modelname + " Results.csv"
    plt.savefig(figfilename)
    plt.close()

    # Creates a CSV of results for the model - contains all data as shown in graphs.
    with open(csvfilename, 'a+', newline='') as file:
        wr = csv.writer(file)
        wr.writerow(y_label)
        wr.writerow(x)
        wr.writerow(y)

# Calculates the accuracy by comparing number of correct and incorrect predictions.
def calculate_accuracy(pred, actual):
    top_pred = pred.argmax(1, keepdim = True)
    correct = top_pred.eq(actual.view_as(top_pred)).sum()
    acc = correct.float() / actual.shape[0]
    return acc