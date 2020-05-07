import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def get_predictions(model, iterator, device):
    model.eval()
    images = []; labels = []; probs = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y_predict = model(x)
            y_probability = F.softmax(y_predict, dim=-1)
            top_pred = y_probability.argmax(1, keepdim=True)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_probability.cpu())

    images = torch.cat(images, dim =0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


def plot_confusion_matrix(labels, pred_labels, classes,modelname):
    plt.ioff()
    fig = plt.figure(figsize = (17, 17))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, classes)
    cm.plot(values_format = 'd', cmap = 'Greens', ax = ax)
    plt.xticks(rotation = 20)
    figfilename = modelname + "confusion.png"
    plt.savefig(figfilename)
    plt.close(fig)

def class_report(pred_labels, data, dp):

    t_images, t_labels = zip(*[(image, label) for image, label in
                               [data[i] for i in range(len(data))]]) 
    t_labels = [i for i in t_labels]

    # list of correct labels, list of predicted labels, dps.
    print(metrics.classification_report(t_labels, pred_labels, digits=dp))