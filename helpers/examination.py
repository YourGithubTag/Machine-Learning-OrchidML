import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from sklearn import decomposition
# from sklearn import manifold
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


def plot_confusion_matrix(labels, pred_labels, classes):
    
    fig = plt.figure(figsize = (17, 17))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, classes)
    cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
    plt.xticks(rotation = 20)