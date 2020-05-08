'''

DISCLAIMER: ALL CODE IS WRITTEN FROM SCRATCH AND MAY CONTAIN SIMILARITIES TO CODE FOUND ONLINE.
            APPROPRIATE ACKNOWLEDGEMENTS HAVE BEEN MADE WHERE NECESSARY.

The AlexNet implementation below is identical to the published paper.
All layers and parameters used have been referenced to their appropriate sections of the report.
Note: Repetition of code (e.g. Conv2d, MaxPool2d etc) have not been commented due to similarities with other comments.

Link: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

'''

import torch
import torch.nn as nn

class AlexNet(nn.Module):

   def __init__(self):
      super(AlexNet, self).__init__()
      self.features = nn.Sequential(
         # Conv. Layer 1.
            # 2D Convolution applied based on Section 3.5 - Overall Architecture.
         nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
         nn.ReLU(),
            # Local Response Normalisation Setup as per Section 3.3 - Local Response Normalization.
         nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),     
            # Max Pooling applied based on Section 3.5 - Overall Architecture.      
         nn.MaxPool2d(kernel_size=3, stride=2), 

         # Conv. Layer 2.
         nn.Conv2d(96, 256, 5, padding=2),  
         nn.ReLU(),
         nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
         nn.MaxPool2d(3, 2),

         # Conv. Layer 3.
         nn.Conv2d(256, 384, 3, padding=1),  
         nn.ReLU(),

         # Conv. Layer 4.
         nn.Conv2d(384, 384, 3, padding=1),  
         nn.ReLU(),

         # Conv. Layer 5.
         nn.Conv2d(384, 256, 3, padding=1), 
         nn.ReLU(),

         nn.MaxPool2d(3, 2)
      )
      
      self.classifier = nn.Sequential(
         # Dropout (zeros elements of some input tensors) based on Section 4.2 - Dropout.
         nn.Dropout(p=0.5, inplace=True),
         # Linear transformation based on Section 3.5 - Overall Architecture.
         nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
         nn.ReLU(),
         nn.Dropout(0.5, True),
         nn.Linear(4096, 4096),
         nn.ReLU(),
         nn.Linear(4096, 17)
      )
   
   # Forward() links all the layers together based on Section 3.5 - Overall Architecture.
   def forward(self, x):
      x = self.features(x)
      x = torch.flatten(x, 1)
      x = self.classifier(x)
      return x