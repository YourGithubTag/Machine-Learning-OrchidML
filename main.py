# Tools
import os
import wget
import shutil
# PyTorch
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
# Math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# Models
from model.alexnet import AlexNet

#---------------- GOOGLE COLAB ----------------#

running_on_google_colab = False
files_downloaded = False

if running_on_google_colab:
   file_path = '/content/flower_data.tar.gz'
   extract_to = '/content/flower_data'
else:
   file_path = './flower_data.tar.gz'
   extract_to = './flower_data'

#---------------- DOWNLOAD DATA ----------------#

if not files_downloaded:
   wget.download('https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz')
   wget.download('https://raw.githubusercontent.com/udacity/pytorch_challenge/master/cat_to_name.json')
   shutil.unpack_archive(file_path, extract_to)
   os.remove(file_path)

#---------------- SAVE IMAGE JPG ----------------#

# Adapted from Example Code - Takes Tensors and converts to RGB image.
def imsave(img):
    npimg = img.numpy()
    npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
    im = Image.fromarray(npimg)
    im.save("./results/your_file.jpeg")

#------------- DEFINING TRANSFORMS --------------#

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