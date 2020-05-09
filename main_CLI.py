import os
import json
import wget
import shutil

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from models.alexnet import AlexNet
from models.vgg import VGG16
from models.vgg_v2 import VGG_v2
from models.resnext import resnext50_32x4d
from helpers.helpers import *
from helpers.examination import *


#-----------------------------------Network Functions-----------------------------------#

def train(model, device, train_loader, validate_loader, optimizer, epoch):
   model.train()
   loss_value = 0
   acc_value = 0
   
   # Trains images from every batch depending on batch size. Calculates loss, accuracy and gradients.
   for index, (inputs, labels) in enumerate(train_loader):
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      output = model(inputs)
      loss = F.cross_entropy(output, labels)
      accuracy = calculate_accuracy(output, labels)
      loss.backward()
      optimizer.step()

      # Sums loss and accuracy for visual results.
      loss_value += loss.item()
      acc_value += accuracy.item()

      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss: {:.4f}\tAccuracy: {:.2f}%'.format(
            epoch, index*len(inputs), len(train_loader.dataset), 100. * index / len(train_loader),
            loss_value / len(train_loader), acc_value / len(train_loader) * 100))
        
   return (loss_value / len(train_loader)), (acc_value / len(train_loader) * 100)       

def evaluate(model, device, evaluate_loader, valid):
   model.eval()
   loss = 0
   accuracy = 0

   # Evaluates images from every batch depending on batch size. Turns off gradient for evaluation purposes.
   # Calculates the loss and accuracy based on evaluation dataset.
   with torch.no_grad():
      for inputs, labels in evaluate_loader:
         inputs, labels = inputs.to(device), labels.to(device)
         output = model.forward(inputs)
         loss += F.cross_entropy(output, labels, reduction='sum').item()
         pred = output.argmax(dim=1, keepdim=True) 
         accuracy += pred.eq(labels.view_as(pred)).sum().item()
   loss /= len(evaluate_loader.dataset)

   if valid:
      word = 'Validate'
   else:
      word = 'Test'

   print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
         word, loss, accuracy, len(evaluate_loader.dataset),
         100. * accuracy / len(evaluate_loader.dataset)))

   return loss, (100. * accuracy / len(evaluate_loader.dataset)) # returns loss and accuracy in %.


#--------------------------------------Main Function--------------------------------------#

def main():
   print(" ██████  ██████   ██████ ██   ██ ██ ██████  ███    ███ ██      ")
   print("██    ██ ██   ██ ██      ██   ██ ██ ██   ██ ████  ████ ██      ")
   print("██    ██ ██████  ██      ███████ ██ ██   ██ ██ ████ ██ ██      ")
   print("██    ██ ██   ██ ██      ██   ██ ██ ██   ██ ██  ██  ██ ██      ")
   print(" ██████  ██   ██  ██████ ██   ██ ██ ██████  ██      ██ ███████ ")

   # Model and epoch selection.
   print("\nSelect a model you would like to train, e.g. 0, 1, 2 or 3: ")
   model_sel = int(input(" 0. AlexNet \n 1. VGG-16  \n 2. VGG_v2  \n 3. ResNext-50 \n> "))
 
   while (model_sel != 0 and model_sel != 1 and model_sel != 2 and model_sel != 3):
      print ("Please input a valid model input.")
      model_sel = int(input(" 0. AlexNet \n 1. VGG-16  \n 2. VGG_v2  \n 3. ResNext-50 \n> "))
   
   epochs = int(input("How many epochs would you like to run? \n> "))
   
   # Defaults - DO NOT CHANGE.
   train_batch_size = 64
   best_valid_loss = float('inf')
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # Accuracy and Loss Lists for Graphing
   x_epochs = list(range(1, epochs+1))
   y_train_acc = []; y_train_loss = []
   y_valid_acc = []; y_valid_loss = []
   y_test_acc = []; y_test_loss = []

   #-----------------------------------Selecting a Model----------------------------------#

   if model_sel == 0:
      model = AlexNet()
      model_name = "AlexNet"
      model_save = 'AlexNet-model.pt'

   elif model_sel == 1:
      model = VGG16()
      model_name = "VGG-16"
      model_save = 'VGG16-model.pt'

   elif model_sel == 2:
      model = VGG_v2()
      model_name = "VGG_v2"
      model_save = 'VGG_v2-model.pt'

   elif model_sel == 3:
      model = resnext50_32x4d()
      model_name = "ResNext-50"
      model_save = 'ResNext50-model.pt'
      train_batch_size = 32 # OR 16

   #-----------------------------------Dataset Download-----------------------------------#

   file_path = './17Flowers.zip'
   extract_to = './flower_data'

   wget.download('https://dl.dropboxusercontent.com/s/7sk2z16uvjzot81/17Flowers.zip')
   wget.download('https://dl.dropboxusercontent.com/s/rwc40rv1r79tl18/cat_to_name.json')
   shutil.unpack_archive(file_path, extract_to)
   os.remove(file_path)
   print('Files have successfully downloaded.')

   #-----------------------------------Data Preparation-----------------------------------#
   
   train_transform = transforms.Compose([
                              transforms.RandomChoice([
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomVerticalFlip(p=0.5),
                                 transforms.RandomRotation(180),
                                 ]),
                              transforms.Resize(256),
                              transforms.CenterCrop(227),
                              transforms.ToTensor()
                     ])

   validate_test_transform = transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(227),
                                       transforms.ToTensor()
                              ])

   # Prepares and Loads Training, Validation and Testing Data.
   train_data = datasets.ImageFolder(extract_to+'/train', transform=train_transform)
   validate_data = datasets.ImageFolder(extract_to+'/valid', transform=validate_test_transform)
   test_data = datasets.ImageFolder(extract_to+'/test', transform=validate_test_transform)

   # Defining the Dataloaders using Datasets.
   train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
   validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=136)         
   test_loader = torch.utils.data.DataLoader(test_data, batch_size=136)                 

   # Compile labels into a list from JSON file.
   with open('cat_to_name.json', 'r') as f:
      cat_to_name = json.load(f)

   species = []
   for label in cat_to_name:
      species.append(cat_to_name[label])

   #---------------------------------Setting up the Network---------------------------------#
  
   model = model.to(device)

   if model_sel == 1:
      optimizer = optim.SGD(model.parameters(), lr=0.001)
   else:
      optimizer = optim.Adam(model.parameters(), lr=0.001)

   print(f'Device selected: {str(device).upper()}')
   print(f'\nNumber of training samples: {len(train_data)}')
   print(f'Number of validation samples: {len(validate_data)}')
   print(f'Number of testing samples: {len(test_data)}')
   
   #----------------------------------Training the Network----------------------------------#
   
   for epoch in range(1, epochs+1):
      train_loss, train_acc = train(model, device, train_loader, validate_loader, optimizer, epoch)
      valid_loss, valid_acc = evaluate(model, device, validate_loader, 1)
      test_loss, test_acc = evaluate(model, device, test_loader, 0)

      # Saves results from training, validation and testing (accuracy and loss) into respective lists.
      y_train_acc.append(round(train_acc, 2)); y_train_loss.append(round(train_loss, 3))
      y_valid_acc.append(round(valid_acc, 2)); y_valid_loss.append(round(valid_loss, 3))
      y_test_acc.append(round(test_acc, 2)); y_test_loss.append(round(test_loss, 3))

      # Saves new version of model based on value of loss from validation.
      if valid_loss < best_valid_loss:
         best_valid_loss = valid_loss
         torch.save(model.state_dict(), model_save)
         print('Current Best Valid Loss: {:.4f}.\n'.format(best_valid_loss))

   #----------------------------------Accuracy/Loss Graphs----------------------------------#
   
   plot_graphs_csv(x_epochs, y_train_acc, ['Train Accuracy'], 'Train Accuracy', model_name)
   plot_graphs_csv(x_epochs, y_train_loss, ['Train Loss'], 'Train Loss', model_name)
   plot_graphs_csv(x_epochs, y_valid_acc, ['Validate Accuracy'], 'Validate Accuracy', model_name)
   plot_graphs_csv(x_epochs, y_valid_loss, ['Validate Loss'], 'Validate Loss', model_name)
   plot_graphs_csv(x_epochs, y_test_acc, ['Test Accuracy'], 'Test Accuracy', model_name)
   plot_graphs_csv(x_epochs, y_test_loss, ['Test Loss'], 'Test Loss', model_name)

   #-----------------------------------Testing the Network-----------------------------------#
  
   model.load_state_dict(torch.load(model_save))
   _, _ = evaluate(model, device, test_loader, 0)

   #---------------------------------Examination of Results----------------------------------#

   # Gathers predictions made by the model for use further below.
   get_predictions(model, test_loader, device)
   _, labels, probs = get_predictions(model, test_loader, device)
   predicted_labels = torch.argmax(probs, 1)

   # Uses data from above to plot confusion matrix, precision rate, recall rate and f1 rate.
   plot_confusion_matrix(labels, predicted_labels, species, model_name)
   class_report(predicted_labels, test_data, 3)

if __name__ == '__main__':
   main()