# Module and Model Imports
import os
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

# Google Colab.
running_on_google_colab = False
files_downloaded = False

if running_on_google_colab:
   zip = '/content/flower_data.tar.gz'
   data = '/content/flower_data'
else:
   zip = './flower_data.tar.gz'
   data = './flower_data'

# Downloads the Dataset.
if not files_downloaded:
   wget.download('https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz')
   wget.download('https://raw.githubusercontent.com/udacity/pytorch_challenge/master/cat_to_name.json')
   shutil.unpack_archive(zip, data)
   os.remove(zip)

# Function: Saves Tensor as Image.
   # Update comments later.
   # Adapted from Example Code - Takes Tensors and converts to RGB image.
def imsave(img):
    npimg = img.numpy()
    npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
    im = Image.fromarray(npimg)
    im.save("./results/your_file.jpeg")

# Transform definitions for Train, Validate and Test.
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