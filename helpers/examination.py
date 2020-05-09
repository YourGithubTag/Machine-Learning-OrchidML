import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Function which collects and returns the images used for testing,
# their labels and the predictions (probabilities).
def get_predictions(model, iterator, device):
    model.eval()
    images = []; labels = []; probs = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y_predict = model(x)
            y_probability = F.softmax(y_predict, dim=-1)
            top_pred = y_probability.argmax(1, keepdim=True)
            # Adds to the lists (using the CPU to reduce strain on GPU memory).
            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_probability.cpu())

    # Concatenates all tensors from each respective list.
    images = torch.cat(images, dim =0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs

# Plots the confusion matrix using SciLearn Kit. Interaction with plots is deactivated
# and is saved in root with their respective model names.
def plot_confusion_matrix(labels, pred_labels, classes, modelname):
    plt.ioff()
    fig = plt.figure(figsize = (17, 17))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, classes)
    cm.plot(values_format = 'd', cmap = 'Greens', ax = ax)
    plt.xticks(rotation = 20)
    figfilename = modelname + " confusion.png"
    plt.savefig(figfilename)
    plt.close(fig)

# Generates the results for precision rate, recall rate and f1 rate.
# Provides accuracy results as well using SciLearn Kit.
def class_report(pred_labels, data, dp):
    
    # Extracts all correct labels from the data_loader.
    t_images, t_labels = zip(*[(image, label) for image, label in
                               [data[i] for i in range(len(data))]]) 
    t_labels = [i for i in t_labels]

    # list of correct labels, list of predicted labels, dps.
    print(metrics.classification_report(t_labels, pred_labels, digits=dp))