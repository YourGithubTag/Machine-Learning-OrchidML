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
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from model.alexnet import AlexNet

# Adapted from Example Code - Takes Tensors and converts to RGB image.
def imsave(img):
   npimg = img.numpy()
   npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
   im = Image.fromarray(npimg)
   im.save("./results/your_file.jpeg")

# Training Function.
def train(model, device, train_loader, optimizer, epoch):
   model.train()
   for batch_index, (inputs, labels) in enumerate(train_loader):
      data, target = data.to(device), target.to(device)
      # zero the parameter gradients
      optimizer.zero_grad()
      # forward + backward + optimize
      output = model(data.view(-1,28*28))
      loss = F.nll_loss(output, target)
      loss.backward(); optimizer.step()
      if batch_idx % log_interval == 0:
         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))

# Validation Function.
def validate():

# Testing Function.
def test():


def main():
   # VARIABLES
   epochs = 10
   gamma = 0.7
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f'Device selected: {str(device).upper()}')

   # GOOGLE COLAB
   running_on_google_colab = False
   files_downloaded = True

   if running_on_google_colab:
      file_path = '/content/flower_data.tar.gz'
      extract_to = '/content/flower_data'
   else:
      file_path = './flower_data.tar.gz'
      extract_to = './flower_data'

   # DOWNLOAD DATA
   if not files_downloaded:
      wget.download('https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz')
      wget.download('https://raw.githubusercontent.com/udacity/pytorch_challenge/master/cat_to_name.json')
      shutil.unpack_archive(file_path, extract_to)
      os.remove(file_path)

   # DEFINING TRANSFORMS
   train_transform = transforms.Compose([
      # Selects a random transform from the list and applies it respective
      # of its corresponding probability.
      transforms.RandomChoice([
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomVerticalFlip(p=0.5),
         transforms.RandomRotation(180),
         ]),
      # Crops image to 256x256.
      transforms.RandomResizedCrop(256),
      # Converts to Tensor.
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])

   validate_test_transform = transforms.Compose([
      # Crops image to 256x256 from centre.
      transforms.CenterCrop(256),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])

   # Prepares and Loads Training, Validation and Testing Data.
   train_data = datasets.ImageFolder(data+'/train', transform=train_transform)
   validate_data = datasets.ImageFolder(data+'/valid', transform=validate_test_transform)
   test_data = datasets.ImageFolder(data+'/test', transform=validate_test_transform)

   # MAPPING CATEGORY LABELS
   with open('./cat_to_name.json', 'r') as ctn:
      cat_to_name = json.load(ctn)

if __name__ == '__main__':
   main()
