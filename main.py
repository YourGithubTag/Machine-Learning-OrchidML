# Module and Model Imports
import os
import json
import wget
import shutil
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from model.alexnet import AlexNet

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

# Adapted from Example Code - Takes Tensors and converts to RGB image.
def imsave(img):
   npimg = img.numpy()
   npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
   im = Image.fromarray(npimg)
   im.save("./results/your_file.jpeg")

# Training Function.
def train(model, device, train_loader, validate_loader, optimizer, epoch):
   model.train()
   for index, (inputs, labels) in enumerate(train_loader):
      # inputs = batch of samples (64) || index = batch index (1)
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      output = model(inputs)
      loss = F.cross_entropy(output, labels)
      loss.backward()
      optimizer.step()

      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
         epoch, index*len(inputs), len(train_loader.dataset), 
         100. * index / len(train_loader), loss.item()))

# Validation and Testing Function.
def valid_test(model, device, valid_test_loader, valid):
   model.eval()
   loss = 0; accuracy = 0
   with torch.no_grad():
      for images, labels in valid_test_loader:
         images, labels = images.to(device), labels.to(device)
         output = model.forward(images)
         loss += F.cross_entropy(output, labels, reduction='sum').item()  # loss is summed before adding to loss
         pred = output.argmax(dim=1, keepdim=True)  # index of the max log-probability
         accuracy += pred.eq(labels.view_as(pred)).sum().item()

   loss /= len(valid_test_loader.dataset)

   if valid:
      word = 'Validate'
   else:
      word = 'Test'

   print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
         word, loss, accuracy, len(valid_test_loader.dataset),
         100. * accuracy / len(valid_test_loader.dataset)))

def main():
   # VARIABLES
   epochs = 14
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f'Device selected: {str(device).upper()}')
   torch.manual_seed(1)

   # GOOGLE COLAB
   running_on_google_colab = False
   files_downloaded = True

   if running_on_google_colab:
      file_path = '/content/17Flowers.zip'
      extract_to = '/content/flower_data'
   else:
      file_path = './17Flowers.zip'
      extract_to = '../flower_data'

   # DOWNLOAD DATA
   if not files_downloaded:
      wget.download('https://dl.dropboxusercontent.com/s/itlaky1ssv8590j/17Flowers.zip')
      wget.download('https://dl.dropboxusercontent.com/s/rwc40rv1r79tl18/cat_to_name.json')
      shutil.unpack_archive(file_path, extract_to)
      os.remove(file_path)
      print('Files Downloaded...')
   
   # MAPPING CATEGORY LABELS
   with open('./cat_to_name.json', 'r') as ctn:
      cat_to_name = json.load(ctn)

   # DEFINING TRANSFORMS
   train_transform = transforms.Compose([
      # Selects a random transform from the list and applies it respective
      # of its corresponding probability.
      transforms.RandomChoice([
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomVerticalFlip(p=0.5),
         transforms.RandomRotation(180),
         ]),
      # Crops image to 227x227.
      transforms.Resize(256),
      transforms.CenterCrop(227),
      # Converts to Tensor.
      transforms.ToTensor()            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])

   validate_test_transform = transforms.Compose([
      # Crops image to 227x227 from centre.
      transforms.Resize(256),
      transforms.CenterCrop(227),
      transforms.ToTensor()            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])

   # Prepares and Loads Training, Validation and Testing Data.
   train_data = datasets.ImageFolder(extract_to+'/train', transform=train_transform)
   validate_data = datasets.ImageFolder(extract_to+'/valid', transform=validate_test_transform)
   test_data = datasets.ImageFolder(extract_to+'/test', transform=validate_test_transform)

   # Plots Training Images
   N_IMAGES = 25
   images, labels = zip(*[(image, label) for image, label in 
                              [train_data[i] for i in range(N_IMAGES)]])
   labels = [test_data.classes[i] for i in labels]
   plot_images(images, labels, normalize = True)
   

   # Defining the Dataloaders using Datasets.
   train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)  # 1,088.
   validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=68)         # 136.
   test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)                 # 136.

   model = AlexNet().to(device)

   optimizer = optim.Adam(model.parameters(), lr=0.001)

   # for epoch in range(1, epochs+1):
   #    train(model, device, train_loader, validate_loader, optimizer, epoch)
   #    valid_test(model, device, test_loader, 0)

if __name__ == '__main__':
   main()