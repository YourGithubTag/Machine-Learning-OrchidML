import os
import json
import shutil
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from model.inception import Inception3
from model.ResNext import resnext50_32x4d
from helpers.helpers import *
from helpers.examination import *

#-----------------------------------Network Functions-----------------------------------#

def train(model, device, train_loader, validate_loader, optimizer, epoch):
   model.train()
   loss_value = 0
   for index, (inputs, labels) in enumerate(train_loader):
      # inputs = batch of samples (64) || index = batch index (1)
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      output = model(inputs)
      loss = F.cross_entropy(output, labels)
      loss_value += loss.item()
      loss.backward()
      optimizer.step()

      if index == 16:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss: {:.6f}'.format(
          epoch, index*len(inputs), len(train_loader.dataset), 
          100. * index / len(train_loader), loss_value / len(train_loader)))

def evaluate(model, device, evaluate_loader, valid):
   model.eval()
   loss = 0
   accuracy = 0
   with torch.no_grad():
      for inputs, labels in evaluate_loader:
         inputs, labels = inputs.to(device), labels.to(device)
         output = model.forward(inputs)
         loss += F.cross_entropy(output, labels, reduction='sum').item()  # loss is summed before adding to loss
         pred = output.argmax(dim=1, keepdim=True)  # index of the max log-probability
         accuracy += pred.eq(labels.view_as(pred)).sum().item()

   loss /= len(evaluate_loader.dataset)

   if valid:
      word = 'Validate'
   else:
      word = 'Test'

   print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
         word, loss, accuracy, len(evaluate_loader.dataset),
         100. * accuracy / len(evaluate_loader.dataset)))

   return loss, (100. * accuracy / len(evaluate_loader.dataset)) # returns loss and accuracy in %.
   
def jobSetup():

   a = True
   b = True
   c = True
   d = True
   e = True
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   while (a):
      collab = input("On google collab?")

      if (collab != 'y' and collab != 'n'):
         print ("Please input a valid collab input")
         a = True

      if (collab == 'y'):
         google_colab = True
         a = False
         print ("collab on")

      if (collab == 'n'):
         google_colab = False
         a = False
         print ("collab off")

   while (b):
      imagesinput = input("Plot images?")
      if (imagesinput != 'y' and imagesinput != 'n'):
         print ("Please input a valid plot input")
         b = True
      if (imagesinput == 'y'):
         imagesplot = True
         b = False
         print ("plot on")

      if (imagesinput == 'n'):
         imagesplot = False
         b = False
         print ("plot off")

   while (c):
      sessiontype = input("From Stratch: a, Continue learning: b, Testing: c") 
      if (sessiontype != 'a' and sessiontype != 'b' and sessiontype != 'c'):
         print ("Please input a valid session input")
         c = True
      if (sessiontype == 'a'):
         c = False
         print ("From Stratch: chosen")
      elif (sessiontype == 'b'):
         c = False
         print ("Continue learning: chosen")
      elif (sessiontype == 'c'):
         c = False
         print ("Testing: chosen")

   while (d):
      modeltype = input("Alexnet: a, AlexnetVGG: b , ResNext: c, Inception: d") 
      if (modeltype != 'a' and modeltype != 'b' and modeltype != 'c' and modeltype != 'd'):
         print ("Please input a valid model input")
         d = True

      if (modeltype == 'a'):
         model = Inception3().to(device)
         modeldict = 'inceptionv3-model.pt'
         optimizer = optim.Adam(model.parameters(), lr=0.001)
         d = False
         print ("Inception3: chosen")

      elif (modeltype == 'b'):
         model = Inception3().to(device)
         modeldict = 'inceptionv3-model.pt'
         optimizer = optim.Adam(model.parameters(), lr=0.001)
         d = False
         print ("Inception3: chosen")

      elif (modeltype == 'c'):
         model = resnext50_32x4d().to(device)
         modeldict = 'ResNext-model.pt'
         optimizer = optim.Adam(model.parameters(), lr=0.001)
         d = False
         print ("ResNext: chosen")

      elif (modeltype == 'd'):
         model = Inception3().to(device)
         modeldict = 'inceptionv3-model.pt'
         optimizer = optim.Adam(model.parameters(), lr=0.001)
         d = False
         print ("Inception3: chosen")

   while (e):
      epoch = input("Number of Epochs: ")
      try:
         val = int(epoch)
         print(f'\nEpochs chosen: {epoch}')
         e = False
      except ValueError:
         print ("Please input a valid model input")
         e = True

   return google_colab, imagesplot, sessiontype,model, modeldict, optimizer, val, device

#--------------------------------------Main Function--------------------------------------#

def main():
   running_on_google_colab, imagesplot, sessiontype, model, modeldict, optimizer, epochs, device  = jobSetup()
   best_valid_loss = float('inf')

   # accuracy and loss graphing
   x_epochs = list(range(1, epochs+1))
   y_train_loss = []
   y_valid_acc = []; y_valid_loss = []
   y_test_acc = []; y_test_loss = []

   #-----------------------------------Dataset Download-----------------------------------#

   files_downloaded = True

   if running_on_google_colab:
      file_path = '/content/17Flowers.zip'
      extract_to = '/content/flower_data'
   else:
      file_path = './17Flowers.zip'
      extract_to = './flower_data'
      print('extracted.')

   if not files_downloaded:
      #wget.download('https://dl.dropboxusercontent.com/s/itlaky1ssv8590j/17Flowers.zip')
      #wget.download('https://dl.dropboxusercontent.com/s/rwc40rv1r79tl18/cat_to_name.json')
      shutil.unpack_archive(file_path, extract_to)
      os.remove(file_path)
      print('Files have successfully downloaded.')
   else:
      print('Files have been downloaded.')

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
   
   if (sessiontype == 'a' or sessiontype == 'b'):
      #Defining the Dataloaders using Datasets.     # 136.
      train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)  # 1,088.
      validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=32,shuffle=True)    
      # Prepares and Loads Training, Validation and Testing Data.
      train_data = datasets.ImageFolder(extract_to+'/train', transform=train_transform)
      validate_data = datasets.ImageFolder(extract_to+'/valid', transform=validate_test_transform)

   test_data = datasets.ImageFolder(extract_to+'/test', transform=validate_test_transform)
   test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)                 # 136.

   # Compile labels into a list from JSON file.
   with open('cat_to_name.json', 'r') as f:
      cat_to_name = json.load(f)

   species = []
   for label in cat_to_name:
      species.append(cat_to_name[label])
   #---------------------------------Plots Training Images---------------------------------#
   if (imagesplot):
      N_IMAGES = 25
      images, labels = zip(*[(image, label) for image, label in 
                                 [train_data[i] for i in range(N_IMAGES)]])
      labels = [test_data.classes[i] for i in labels]
      plot_images(images, labels, normalize = True)
   

   #---------------------------------Setting up the Network---------------------------------#
   
   
   print(f'Device selected: {str(device).upper()}')
   print(f'\nNumber of training samples: {len(train_data)}')
   print(f'Number of validation samples: {len(validate_data)}')
   print(f'Number of testing samples: {len(test_data)}')

   #----------------------------------Training the Network----------------------------------#
   sessionplotting = False

   if (sessiontype == 'a'):
      for epoch in range(1, epochs+1):
         train_loss = train(model, device, train_loader, validate_loader, optimizer, epoch)
         valid_loss, valid_acc = evaluate(model, device, validate_loader, 1)
         test_loss, test_acc = evaluate(model, device, test_loader, 0)
         
         y_train_loss.append(round(train_loss,3))
         y_valid_acc.append(round(valid_acc, 0)); y_valid_loss.append(round(valid_loss, 3))
         y_test_acc.append(round(test_acc, 0)); y_test_loss.append(round(test_loss, 3))
         
         if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), modeldict)
            print('Current Best Valid Loss: {:.4f}.\n'.format(best_valid_loss))
      sessionplotting = True
      sessiontype = 'c'

   elif (sessiontype == 'b'): 
      model.load_state_dict(torch.load(modeldict))	
      for epoch in range(1, epochs+1):
         train_loss = train(model, device, train_loader, validate_loader, optimizer, epoch)
         valid_loss, valid_acc = evaluate(model, device, validate_loader, 1)
         test_loss, test_acc = evaluate(model, device, test_loader, 0)
         
         y_train_loss.append(round(train_loss,3))
         y_valid_acc.append(round(valid_acc, 0)); y_valid_loss.append(round(valid_loss, 3))
         y_test_acc.append(round(test_acc, 0)); y_test_loss.append(round(test_loss, 3))
         
         if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), modeldict)
            print('Current Best Valid Loss: {:.4f}.\n'.format(best_valid_loss))
      sessionplotting = True
      sessiontype = 'c'

   if (sessiontype == 'c'):
      model.load_state_dict(torch.load(modeldict))
      _, _ = evaluate(model, device, test_loader, 0)

   if (sessionplotting):
      plot_graphs_csv(x_epochs, y_train_loss, ['Train Loss'])
      plot_graphs_csv(x_epochs, y_valid_acc, ['Validate Accuracy'])
      plot_graphs_csv(x_epochs, y_valid_loss, ['Validate Loss'])
      plot_graphs_csv(x_epochs, y_test_acc, ['Test Accuracy'])
      plot_graphs_csv(x_epochs, y_test_loss, ['Test Loss'])

      get_predictions(model, test_loader, device)
      images, labels, probs = get_predictions(model, test_loader, device)
      predicted_labels = torch.argmax(probs, 1)

      plot_confusion_matrix(labels, predicted_labels, species)
      class_report(predicted_labels, test_data, 3)


if __name__ == '__main__':
   main()
