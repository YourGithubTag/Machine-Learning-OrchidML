import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image

def plot_images(images, labels, normalize = False):
    n_images = len(images)
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    fig = plt.figure(figsize = (10, 10))

    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        image = images[i]
        if normalize:
            image_min = image.min()
            image_max = image.max()
            image.clamp_(min = image_min, max = image_max)
            image.add_(-image_min).div_(image_max - image_min + 1e-5)
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title(labels[i])
        ax.axis('off')
    
    plt.show()

def plot_graphs_csv(x, y, y_label):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)

    with open('Results.csv', 'a+', newline='') as file:
        wr = csv.writer(file)
        wr.writerow(y_label)
        wr.writerow(x)
        wr.writerow(y)