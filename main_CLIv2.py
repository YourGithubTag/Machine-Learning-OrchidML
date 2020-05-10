import os
import json
import shutil
import wget
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from models.vgg_v2 import VGG_v2
from model.vgg import VGG16
from model.alexnet import AlexNet
from model.resnext import resnext50_32x4d
from helpers.helpers import *
from helpers.examination import *

#-----------------------------------Network Functions-----------------------------------#

def train(model, device, train_loader, validate_loader, optimizer, epoch):
   model.train()
   loss_value = 0
   acc_value = 0

    # Trains images from every batch depending on batch size. Calculates loss, accuracy and gradients.
   for index, (inputs, labels) in enumerate(train_loader):
      # inputs = batch of samples (64) || index = batch index (1)
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
   def __init__(self,sessiontype,model, modeldict, optimizer, epochval, device, trainbatch,testbatch,modelname):
    self.sessiontype = sessiontype
    self.model = model
    self.modeldict = modeldict
    self.optimizer = optimizer
    self.epochs = epochval
    self.device = device
    self.trainbatch = trainbatch
    self.testbatch = testbatch
    self.modelname = modelname


def typeface():
   print(" ██████  ██████   ██████ ██   ██ ██ ██████  ███    ███ ██      ")
   print("██    ██ ██   ██ ██      ██   ██ ██ ██   ██ ████  ████ ██      ")
   print("██    ██ ██████  ██      ███████ ██ ██   ██ ██ ████ ██ ██      ")
   print("██    ██ ██   ██ ██      ██   ██ ██ ██   ██ ██  ██  ██ ██      ")
   print(" ██████  ██   ██  ██████ ██   ██ ██ ██████  ██      ██ ███████ ")



#------------------------------------Job Creation---------------------------------#
def jobSetup():
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   exit = False # Exit jobsetup Boolean
   joblist = []
   while (not exit):
      # These booleans control the state of the menu
      SessionTypeBool = True
      ModelTypeBool = True
      EpochBool = True
      TrainBatchBool = True
      OptimBool = True
      TestBatchBool = True
      jobBool = True


      #--------------------------------------Model Selection--------------------------------------#
      while (ModelTypeBool):
         modeltype = input(" a.Alexnet \n b.VGG16  \n c.ResNext  \n d.VGGv2\n   >") 
         if (modeltype != 'a' and modeltype != 'b' and modeltype != 'c' and modeltype != 'd'):
            print ("Please input a valid model input")
            ModelTypeBool = True

         if (modeltype == 'a'):
            model = AlexNet()
            modeldict = 'Alexnet-model.pt'
            modelname ="Alexnet"
            valtrain = 64
            valtest = 136
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            ModelTypeBool = False

         elif (modeltype == 'b'):
            model = VGG16()
            modeldict = 'VGG16-model.pt'
            modelname ="VGG16"
            valtrain = 32
            valtest = 136
            optimizer = optim.SGD(model.parameters(), lr=0.001)
            ModelTypeBool = False

         elif (modeltype == 'c'):
            model = resnext50_32x4d()
            modeldict = 'ResNext50-model.pt'
            modelname ="ResNext50"
            valtrain = 32
            valtest = 136
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            ModelTypeBool = False

         elif (modeltype == 'd'):
            model = VGG_v2()
            modeldict = 'VGGv2-model.pt'
            modelname ="VGGv2"
            valtrain = 64
            valtest = 136
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            ModelTypeBool = False
      
      print (modelname + ": chosen")

      #------------------------------------Session Selection--------------------------------------#
      while (SessionTypeBool):
         sessiontype = input(" a.Start Training a new model \n b.Test the model \n   >") 
         if (sessiontype != 'a' and sessiontype != 'b' and sessiontype != 'c'):
            print ("Please input a valid session input")
            SessionTypeBool = True
         if (sessiontype == 'a'):
            SessionTypeBool = False
            print ("From Stratch: chosen")
         elif (sessiontype == 'b'):
            SessionTypeBool = False
            TrainBatchBool = False
            OptimBool = False
            EpochBool = False
            valtrain = 1
            epochval = 1
            print ("Testing: chosen") 
        #UNCOMMENT FOR CONTINUE TRAINING OPTION Uncomment and use at your own risk!
         """
         elif (sessiontype == 'c'):
            SessionTypeBool = False
            print ("Testing: chosen")
         """
      #------------------------------------Epoch Selection--------------------------------------#
      while (EpochBool):
         epoch = input(" Number of Epochs:   ")
         try:
            epochval = int(epoch)
            print(f'\nEpochs chosen: {epochval}')
            EpochBool = False
         except ValueError:
            print ("Please input a valid Epochs input")
            EpochBool = True

      # This section is DEVELOPER USE ONLY. We do not want the user to change the training or test batch numbers
      # as this can lead to CUDA out of memory errors. Uncomment and use at your own risk!
      """
      #------------------------------------Optimiser Selection---------------------------------#
      while (OptimBool):
         optimiseinput = input(" Optimizer (Debug): \n a.Adam \n b.SGD  \n   >") 
         if (optimiseinput != 'a' and optimiseinput != 'b'):
            print ("Please input a valid Optimizer input")
            OptimBool = True
         if (optimiseinput == 'a'):  
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            print ("Adam chosen")
            OptimBool = False
         elif (optimiseinput == 'b'):
            optimizer = optim.SGD(model.parameters(), lr=0.001)
            print ("SGD chosen")
            OptimBool = False
      #------------------------------------Batch Selection---------------------------------#
      while (TrainBatchBool):
         trainbatch = input(" Number of train batchs (Debug):   ")
         try:
            valtrain = int(trainbatch)
            print(f'\ntraining batchs chosen: {valtrain}')
            TrainBatchBool = False
         except ValueError:
            print ("Please input a valid batchs input")
            TrainBatchBool = True

      while (TestBatchBool):
         testbatch = input(" Number of test batchs (Debug):   ")
         try:
            valtest = int(testbatch)
            print(f'\ntest batchs chosen: {valtest}')
            TestBatchBool = False
         except ValueError:
            print ("Please input a valid batchs input")
            TestBatchBool = True
      """
      #------------------------------------Job Menu---------------------------------#
      job = jobclass(sessiontype,model, modeldict, optimizer, epochval, device,valtrain,valtest, modelname)
      joblist.append(job)

      while (jobBool):
         finish = input(" Would you like to run another Model after? y/n:   ")
         if (finish != 'y' and finish != 'n'):
            print ("Please input a valid job input")
            jobBool = True
         if (finish == 'y'):
            jobBool = False
            print ("Add another job")

         if (finish == 'n'):
            jobBool = False
            exit = True
            print ("Jobs Executing")
   return joblist

#--------------------------------------Main Function--------------------------------------#

def main():
   typeface() #Intro face
   joblist = jobSetup() # Setting up all jobs to be run
   for currentjob in joblist:

      best_valid_loss = float('inf')

      # accuracy and loss graphing
      x_epochs = list(range(1, currentjob.epochs+1))
      y_train_acc = []; y_train_loss = []
      y_valid_acc = []; y_valid_loss = []
      y_test_acc = []; y_test_loss = []

      #-----------------------------------Dataset Download-----------------------------------#

      file_path = './17Flowers.zip'
      extract_to = './flower_data'
      files_downloaded = False
      
      if not files_downloaded:
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
      
      if (currentjob.sessiontype == 'a' or currentjob.sessiontype == 'c'):
         # Prepares and Loads Training, Validation and Testing Data.
         train_data = datasets.ImageFolder(extract_to+'/train', transform=train_transform)
         validate_data = datasets.ImageFolder(extract_to+'/valid', transform=validate_test_transform)
         #Defining the Dataloaders using Datasets
         train_loader = torch.utils.data.DataLoader(train_data, batch_size=currentjob.trainbatch, shuffle=True)
         validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=currentjob.testbatch,shuffle=True)    

         print(f'\nNumber of training samples: {len(train_data)}')
         print(f'Number of validation samples: {len(validate_data)}')


      test_data = datasets.ImageFolder(extract_to+'/test', transform=validate_test_transform)
      test_loader = torch.utils.data.DataLoader(test_data, batch_size=currentjob.testbatch, shuffle=True)

      # Compile labels into a list from JSON file.
      with open('cat_to_name.json', 'r') as f:
         cat_to_name = json.load(f)

      species = []
      for label in cat_to_name:
         species.append(cat_to_name[label])
   

      #---------------------------------Setting up the Job-------------------------------------#
      print(f'Device selected: {str(currentjob.device).upper()}')
      print(f'Number of testing samples: {len(test_data)}')

      model = currentjob.model.to(currentjob.device)
      sessionplotting = False # Turns on or off the function call to plot all graphs

      #----------------------------------Training the Network----------------------------------#
      if (currentjob.sessiontype == 'a'):

         for epoch in range(1, currentjob.epochs+1):
            train_loss, train_acc = train(model, currentjob.device, train_loader, validate_loader, currentjob.optimizer, epoch)
            valid_loss, valid_acc = evaluate(model, currentjob.device, validate_loader, 1)
            test_loss, test_acc = evaluate(model, currentjob.device, test_loader, 0)
            
            # Saves results from training, validation and testing (accuracy and loss) into respective lists.
            y_train_acc.append(round(train_acc, 2)); y_train_loss.append(round(train_loss, 3))
            y_valid_acc.append(round(valid_acc, 2)); y_valid_loss.append(round(valid_loss, 3))
            y_test_acc.append(round(test_acc, 2)); y_test_loss.append(round(test_loss, 3))
                  
            # Saves new version of model based on value of loss from validation.      
            if valid_loss < best_valid_loss:
               best_valid_loss = valid_loss
               torch.save(model.state_dict(), currentjob.modeldict)
               print('Current Best Valid Loss: {:.4f}.\n'.format(best_valid_loss))
         sessionplotting = True
         currentjob.sessiontype = 'b'

      #-----------------------------------Continue from Save Training-----------------------------#
      elif (currentjob.sessiontype == 'c'): 

         model.load_state_dict(torch.load(currentjob.modeldict))	

         for epoch in range(1, currentjob.epochs+1):
            train_loss, train_acc = train(model, currentjob.device, train_loader, validate_loader, currentjob.optimizer, epoch)
            valid_loss, valid_acc = evaluate(model, currentjob.device, validate_loader, 1)
            test_loss, test_acc = evaluate(model, currentjob.device, test_loader, 0)
            
            # Saves results from training, validation and testing (accuracy and loss) into respective lists.
            y_train_acc.append(round(train_acc, 2)); y_train_loss.append(round(train_loss, 3))
            y_valid_acc.append(round(valid_acc, 2)); y_valid_loss.append(round(valid_loss, 3))
            y_test_acc.append(round(test_acc, 2)); y_test_loss.append(round(test_loss, 3))
                  
            # Saves new version of model based on value of loss from validation.
            if valid_loss < best_valid_loss:
               best_valid_loss = valid_loss
               torch.save(model.state_dict(), currentjob.modeldict)
               print('Current Best Valid Loss: {:.4f}.\n'.format(best_valid_loss))
         sessionplotting = True
         currentjob.sessiontype = 'b'

      #-----------------------------------Testing the Network-----------------------------------#
      if (currentjob.sessiontype == 'b'):
         print("TEST RESULTS: " + currentjob.modelname)
         model.load_state_dict(torch.load(currentjob.modeldict))
         _, _ = evaluate(model, currentjob.device, test_loader, 0)

      #---------------------------------Examination of Results----------------------------------#
      if (sessionplotting):

         #Plots all data points collected during training
         plot_graphs_csv(x_epochs, y_train_acc, ['Train Accuracy'],'Train Accuracy',currentjob.modelname)
         plot_graphs_csv(x_epochs, y_train_loss, ['Train Loss'],'Train Loss',currentjob.modelname)
         plot_graphs_csv(x_epochs, y_valid_acc, ['Validate Accuracy'],'Validate Accuracy',currentjob.modelname)
         plot_graphs_csv(x_epochs, y_valid_loss, ['Validate Loss'],'Validate Loss',currentjob.modelname)
         plot_graphs_csv(x_epochs, y_test_acc, ['Test Accuracy'],'Test Accuracy',currentjob.modelname)
         plot_graphs_csv(x_epochs, y_test_loss, ['Test Loss'],'Test Loss',currentjob.modelname)
         
         # Gathers predictions made by the model for use further below.
         get_predictions(model, test_loader, currentjob.device)
         images, labels, probs = get_predictions(model, test_loader, currentjob.device)
         predicted_labels = torch.argmax(probs, 1)

         # Uses data from above to plot confusion matrix, precision rate, recall rate and f1 rate.
         plot_confusion_matrix(labels, predicted_labels, species, currentjob.modelname)
         class_report(predicted_labels, test_data, 3)

      #---------------------------------Memory Release---------------------------------------#
      #Del section to make sure that memory is released from the GPU or CPU
      if (currentjob.sessiontype == 'a' or currentjob.sessiontype == 'c'):
         del train_data 
         del validate_data 
         del train_loader
         del validate_loader 

      #Memory Release
      del test_data
      del test_loader
      del model
      del currentjob
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()

   typeface() # exit Screen

if __name__ == '__main__':
   main()
