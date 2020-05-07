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
   acc_value = 0
   for index, (inputs, labels) in enumerate(train_loader):
      # inputs = batch of samples (64) || index = batch index (1)
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      output = model(inputs)
      loss = F.cross_entropy(output, labels)
      accuracy = calculate_accuracy(output, labels)
      loss.backward()
      optimizer.step()

      loss_value += loss.item()
      acc_value += accuracy.item()

      if index == 16:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss: {:.4f}\tAccuracy: {:.2f}'.format(
          epoch, index*len(inputs), len(train_loader.dataset), 
          100. * index / len(train_loader), loss_value / len(train_loader), acc_value / len(train_loader) * 100))
        
   return (loss_value / len(train_loader)), (acc_value / len(train_loader) * 100)

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

class jobclass():
   def __init__(self, google_colab, imagesplot, sessiontype,model, modeldict, optimizer, epochval, device, trainbatch,testbatch,modelname):
    self.google_colab = google_colab
    self.imagesplot = imagesplot
    self.sessiontype = sessiontype
    self.model = model
    self.modeldict = modeldict
    self.optimizer = optimizer
    self.epochs = epochval
    self.device = device
    self.trainbatch = trainbatch
    self.testbatch = testbatch
    self.modelname = modelname

def jobSetup():
   exit = False
   joblist = []
   while (not exit):
      a = True
      b = True
      c = True
      d = True
      e = True
      f = True
      g = True
      h = True
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
            e = False
            f = False
            valtrain = 1
            epochval = 1
            print ("Testing: chosen")

      while (d):
         modeltype = input("Alexnet: a, AlexnetVGG: b , ResNext: c, Inception: d") 
         if (modeltype != 'a' and modeltype != 'b' and modeltype != 'c' and modeltype != 'd'):
            print ("Please input a valid model input")
            d = True

         if (modeltype == 'a'):
            model = Inception3().to(device)
            modeldict = 'inceptionv3-model.pt'
            modelname ="Alexnet"
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            d = False

         elif (modeltype == 'b'):
            model = Inception3().to(device)
            modeldict = 'inceptionv3-model.pt'
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            modelname ="VGG"
            d = False

         elif (modeltype == 'c'):
            model = resnext50_32x4d().to(device)
            modeldict = 'ResNext-model.pt'
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            modelname ="resnet"
            d = False

         elif (modeltype == 'd'):
            model = Inception3().to(device)
            modeldict = 'inceptionv3-model.pt'
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            modelname ="Inception"
            d = False

      print (modelname + ": chosen")

      while (e):
         epoch = input("Number of Epochs: ")
         try:
            epochval = int(epoch)
            print(f'\nEpochs chosen: {epochval}')
            e = False
         except ValueError:
            print ("Please input a valid model input")
            e = True

      while (f):
         trainbatch = input("Number of train batchs: ")
         try:
            valtrain = int(trainbatch)
            print(f'\ntraining batchs chosen: {valtrain}')
            f = False
         except ValueError:
            print ("Please input a valid model input")
            f = True

      while (g):
         testbatch = input("Number of test batchs: ")
         try:
            valtest = int(testbatch)
            print(f'\ntest batchs chosen: {valtest}')
            g = False
         except ValueError:
            print ("Please input a valid model input")
            g = True

      job = jobclass(google_colab, imagesplot, sessiontype,model, modeldict, optimizer, epochval, device,valtrain,valtest, modelname)
      joblist.append(job)

      while (h):
         finish = input("would you like to add another job? y/n: ")
         if (finish != 'y' and finish != 'n'):
            print ("Please input a valid plot input")
            h = True
         if (finish == 'y'):
            h = False
            print ("Add another job")

         if (finish == 'n'):
            h = False
            exit = True
            print ("Finished")
   return joblist

#--------------------------------------Main Function--------------------------------------#

def main():
   joblist = jobSetup()
   for x in joblist:
      best_valid_loss = float('inf')

      # accuracy and loss graphing
      x_epochs = list(range(1, x.epochs+1))
      y_train_acc = []; y_train_loss = []
      y_valid_acc = []; y_valid_loss = []
      y_test_acc = []; y_test_loss = []

      #-----------------------------------Dataset Download-----------------------------------#

      files_downloaded = True

      if x.google_colab:
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
      
      if (x.sessiontype == 'a' or x.sessiontype == 'b'):
          # Prepares and Loads Training, Validation and Testing Data.
         train_data = datasets.ImageFolder(extract_to+'/train', transform=train_transform)
         validate_data = datasets.ImageFolder(extract_to+'/valid', transform=validate_test_transform)
         #Defining the Dataloaders using Datasets.     # 136.
         train_loader = torch.utils.data.DataLoader(train_data, batch_size=x.trainbatch, shuffle=True)  # 1,088.
         validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=x.testbatch,shuffle=True)    

         print(f'\nNumber of training samples: {len(train_data)}')
         print(f'Number of validation samples: {len(validate_data)}')

      test_data = datasets.ImageFolder(extract_to+'/test', transform=validate_test_transform)
      test_loader = torch.utils.data.DataLoader(test_data, batch_size=x.testbatch, shuffle=True)                 # 136.

      # Compile labels into a list from JSON file.
      with open('cat_to_name.json', 'r') as f:
         cat_to_name = json.load(f)

      species = []
      for label in cat_to_name:
         species.append(cat_to_name[label])
      #---------------------------------Plots Training Images---------------------------------#
      if (x.imagesplot):
         N_IMAGES = 25
         images, labels = zip(*[(image, label) for image, label in 
                                    [train_data[i] for i in range(N_IMAGES)]])
         labels = [test_data.classes[i] for i in labels]
         plot_images(images, labels, normalize = True)
      

      #---------------------------------Setting up the Network---------------------------------#
      
      
      print(f'Device selected: {str(x.device).upper()}')
      print(f'Number of testing samples: {len(test_data)}')

      #----------------------------------Training the Network----------------------------------#
      sessionplotting = False

      if (x.sessiontype == 'a'):

         for epoch in range(1, x.epochs+1):
            train_loss, train_acc = train(x.model, x.device, train_loader, validate_loader, x.optimizer, epoch)
            valid_loss, valid_acc = evaluate(x.model, x.device, validate_loader, 1)
            test_loss, test_acc = evaluate(x.model, x.device, test_loader, 0)
            
            y_train_acc.append(round(train_acc, 2)); y_train_loss.append(round(train_loss, 3))
            y_valid_acc.append(round(valid_acc, 2)); y_valid_loss.append(round(valid_loss, 3))
            y_test_acc.append(round(test_acc, 2)); y_test_loss.append(round(test_loss, 3))
                  
            if valid_loss < best_valid_loss:
               best_valid_loss = valid_loss
               torch.save(x.model.state_dict(), x.modeldict)
               print('Current Best Valid Loss: {:.4f}.\n'.format(best_valid_loss))
         sessionplotting = True
         x.sessiontype = 'c'

      elif (x.sessiontype == 'b'): 

         x.model.load_state_dict(torch.load(x.modeldict))	

         for epoch in range(1, x.epochs+1):
            train_loss, train_acc = train(x.model, x.device, train_loader, validate_loader, x.optimizer, epoch)
            valid_loss, valid_acc = evaluate(x.model, x.device, validate_loader, 1)
            test_loss, test_acc = evaluate(x.model, x.device, test_loader, 0)
            
            y_train_acc.append(round(train_acc, 2)); y_train_loss.append(round(train_loss, 3))
            y_valid_acc.append(round(valid_acc, 0)); y_valid_loss.append(round(valid_loss, 3))
            y_test_acc.append(round(test_acc, 0)); y_test_loss.append(round(test_loss, 3))
            
            if valid_loss < best_valid_loss:
               best_valid_loss = valid_loss
               torch.save(x.model.state_dict(), x.modeldict)
               print('Current Best Valid Loss: {:.4f}.\n'.format(best_valid_loss))
         sessionplotting = True
         x.sessiontype = 'c'

      if (x.sessiontype == 'c'):

         x.model.load_state_dict(torch.load(x.modeldict))
         _, _ = evaluate(x.model, x.device, test_loader, 0)

      if (sessionplotting):
         plot_graphs_csv(x_epochs, y_train_acc, ['Train Accuracy'],'Train Accuracy',x.modelname)
         plot_graphs_csv(x_epochs, y_train_loss, ['Train Loss'],'Train Loss',x.modelname)
         plot_graphs_csv(x_epochs, y_valid_acc, ['Validate Accuracy'],'Validate Accuracy',x.modelname)
         plot_graphs_csv(x_epochs, y_valid_loss, ['Validate Loss'],'Validate Loss',x.modelname)
         plot_graphs_csv(x_epochs, y_test_acc, ['Test Accuracy'],'Test Accuracy',x.modelname)
         plot_graphs_csv(x_epochs, y_test_loss, ['Test Loss'],'Test Loss',x.modelname)

         get_predictions(x.model, test_loader, x.device)
         images, labels, probs = get_predictions(x.model, test_loader, x.device)
         predicted_labels = torch.argmax(probs, 1)

         plot_confusion_matrix(labels, predicted_labels, species, x.modelname)
         class_report(predicted_labels, test_data, 3)


if __name__ == '__main__':
   main()
