import torch
import torch.nn as nn
import torch.nn.functional as F


# This function generates the "bottleneck" block that is used in ResNext. The bottleneck block is the building block which ResNext as described in papers such as 
# [Deep Residual Learning for Image Recognition. 10 Dec 2015, Kaiming He, et al]
# [Aggregated Residual Transformations for Deep Neural Networks. 11 Apr 2017, Saining Xie, et al]

# Stride in original bottleneck is placed in the Convolution with kernal size of 1 (1x1 conv) however as per 
# [ResNet v1.5 for PyTorch, Nvidia Research] https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
# stride is placed in the convolution of kernel size = 3, improving accuracy as described by Nvidia

class Bottleneck(nn.Module):
        expansion = 4 # This value determines by how much each layer increases their filter size by
        def __init__(self,inplanes, planes, stride=1, down=False ,groups=1, base_width=64):
            super(Bottleneck, self).__init__()

            width = int(planes * (base_width / 64.)) * groups # This determines the width at which some of the filters operate
            self.expandedplanes = planes * self.expansion # Expanding the planes number to allow output to the next layer size
            
            # This method contains the operations which are undertaken in the bottleneck architecture
            self.bottleneckseq = nn.Sequential(
                nn.Conv2d(inplanes, width, kernel_size=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                # Convolutions are grouped with a cardinality of 32
                nn.Conv2d(width, width, kernel_size=3, stride=stride,padding=1, groups=groups, bias=False), 
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, self.expandedplane, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.expandedplane))

            self.relu = nn.ReLU(inplace=True)
            self.stride = stride
            self.down = down

            # This is the downsampling function, which allows the input to be added to the output when the input does not 
            # match the output
            self.downsampling  = nn.Sequential(
                nn.Conv2d(inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion))

        def forward(self, x):
            identity = x
            out = self.bottleneckseq(x)

            # Often when the bottleneck is a block on the input to a new layer
            # the input must be downsampled before it can be added back to the output
            if self.down is True:
                identity = self.downsampling(x)

            out += identity
            out = self.relu(out)

            return out


# This function constructs the ResNext network and also contains the forward pass in the macro scope of the network, as described by
# [Deep Residual Learning for Image Recognition. 10 Dec 2015, Kaiming He, et al] and 
# [Aggregated Residual Transformations for Deep Neural Networks. 11 Apr 2017, Saining Xie, et al]
# It uses the Bottleneck building block, which contains the Bottleneck architecture on which ResNext is built off.  
class ResNext(nn.Module):
    def __init__(self,groups,width_per_group):
        super(ResNext, self).__init__()

        # Definitions which are used for the input and output sizes in the hidden layers of the network
        self.inplanes = 64
        self.planeslist = [64,128,156,512] #Contains the matrix size for each layer

        # Groups is an important attribute, which dicates how many grouped convolutions occur in a bottleneck block
        # This is an important attribute which allows ResNext to operate
        self.groups = groups 
        self.base_width = width_per_group

        # Input layer into the function, which conducts convolution, batchnorm and a maxpool before the bottleneck layers
        self.inputLayer = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Constructs the layers, which constains the BottleNecks blocks sequentially 
        self.bottleneckoutput = self.bottlenecklayer()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, 17)
        # Initializing all of the initial weights of the modules in the network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # This function constructs the Bottleneck layering that is applied to a ResNext. It layers the different Bottlenecks and inputs the input sizes
    # Input sizes are calculated ahead of time, as given by the planeslist requirements. 
    # 
    def bottlenecklayer(self, stride=1):
        # The bottleneck layers follow the predetermined (3, 4, 6, 3) architecture which has been described in the 
        # Residual learning for image recognition paper for ResNext50
        return nn.Sequential(

            # Layer 1 - 3 blocks [input:64, output:256]
            Bottleneck(64, self.planeslist[0], stride, down=True, groups=self.groups,base_width=self.base_width),
            Bottleneck(256,self.planeslist[0], groups=self.groups,base_width=self.base_width),
            Bottleneck(256,self.planeslist[0], groups=self.groups,base_width=self.base_width),

            # Layer 2 - 4 blocks [input:256, output:512]
            Bottleneck(256, self.planeslist[1], stride=2,down=True, groups=self.groups,base_width=self.base_width),
            Bottleneck(512, self.planeslist[1], groups=self.groups,base_width=self.base_width),
            Bottleneck(512, self.planeslist[1], groups=self.groups,base_width=self.base_width),
            Bottleneck(512, self.planeslist[1], groups=self.groups,base_width=self.base_width),
            
            # Layer 3 - 6 blocks [input:512, output:624]
            Bottleneck(512, self.planeslist[2], stride=2, down=True, groups=self.groups,base_width=self.base_width),
            Bottleneck(624, self.planeslist[2], groups=self.groups,base_width=self.base_width),
            Bottleneck(624, self.planeslist[2], groups=self.groups,base_width=self.base_width),
            Bottleneck(624, self.planeslist[2], groups=self.groups,base_width=self.base_width),
            Bottleneck(624, self.planeslist[2], groups=self.groups,base_width=self.base_width),
            Bottleneck(624, self.planeslist[2], groups=self.groups,base_width=self.base_width),

            # Layer 4 - 3 blocks [input:624, output:2048]
            Bottleneck(624, self.planeslist[3], stride=2,down=True, groups=self.groups,base_width=self.base_width),
            Bottleneck(2048,self.planeslist[3], groups=self.groups,base_width=self.base_width),
            Bottleneck(2048,self.planeslist[3], groups=self.groups,base_width=self.base_width))

    def forward(self, x):

        x = self.inputLayer(x)

        # Layers  
        x = self.bottleneckoutput(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) # Fully Connected layer

        return x
    
def resnext50_32x4d():
    return ResNext(32 , 4)

