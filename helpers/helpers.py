import matplotlib.pyplot as plt
import numpy as np
import csv


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

    with open(csvfilename, 'a+', newline='') as file:
        wr = csv.writer(file)
        wr.writerow(y_label)
        wr.writerow(x)
        wr.writerow(y)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
