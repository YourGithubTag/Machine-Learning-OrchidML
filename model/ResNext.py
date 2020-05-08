import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# This function generates the "bottleneck" block that is used in ResNext. The bottleneck block is the building block which ResNext is built off.
# It is constructed by taking the input of a base width and planes number and then  
class Bottleneck(nn.Module):
        expansion = 4
        def __init__(self,inplanes, planes, stride=1, down=False ,groups=1, base_width=64):
            super(Bottleneck, self).__init__()
            width = int(planes * (base_width / 64.)) * groups
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            self.bn1 = nn.BatchNorm2d(width)
            self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, groups=groups, bias=False, dilation=dilation)
            self.bn2 = nn.BatchNorm2d(width)
            self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.stride = stride
            self.down = down
            self.downsample  = nn.Sequential(
                conv1x1(inplanes, planes * Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * Bottleneck.expansion))

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.down is True:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out


# This function constructs the ResNext network and also contains the forward pass in the macro scope of the network.
# It does so with use of the make_layer method defined in the class, the bottleneck function, and the inbuilt torch nueral network functions.   
# 

class ResNext(nn.Module):
    def __init__(self,groups,width_per_group,num_classes=1000, zero_init_residual=False):
        super(ResNext, self).__init__()

        self.inplanes = 64
        self.planeslist = [64,128,156,512] #Contains the matrix size for each layer
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Constructs the layers, which constains the BottleNecks blocks sequentially 
        self.layer1out = self.layer1()
        self.layer2out = self.layer2(stride=2)
        self.layer3out = self.layer3(stride=2)
        self.layer4out = self.layer4(stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def layer1(self, stride=1):
        return nn.Sequential(
            Bottleneck(64, self.planeslist[0], stride, down=True, groups=self.groups,base_width=self.base_width),
            Bottleneck(256,self.planeslist[0], groups=self.groups,base_width=self.base_width),
            Bottleneck(256,self.planeslist[0], groups=self.groups,base_width=self.base_width))

    def layer2(self, stride=1):
        return nn.Sequential(
            Bottleneck(256, self.planeslist[1], stride,down=True, groups=self.groups,base_width=self.base_width),
            Bottleneck(512, self.planeslist[1], groups=self.groups,base_width=self.base_width),
            Bottleneck(512, self.planeslist[1], groups=self.groups,base_width=self.base_width),
            Bottleneck(512, self.planeslist[1], groups=self.groups,base_width=self.base_width))

    def layer3(self, stride=1):
         return nn.Sequential(
            Bottleneck(512, self.planeslist[2], stride, down=True, groups=self.groups,base_width=self.base_width),
            Bottleneck(624, self.planeslist[2], groups=self.groups,base_width=self.base_width),
            Bottleneck(624, self.planeslist[2], groups=self.groups,base_width=self.base_width),
            Bottleneck(624, self.planeslist[2], groups=self.groups,base_width=self.base_width),
            Bottleneck(624, self.planeslist[2], groups=self.groups,base_width=self.base_width),
            Bottleneck(624, self.planeslist[2], groups=self.groups,base_width=self.base_width))

    def layer4(self, stride=1):
        return nn.Sequential(
            Bottleneck(624, self.planeslist[3], stride,down=True, groups=self.groups,base_width=self.base_width),
            Bottleneck(2048,self.planeslist[3], groups=self.groups,base_width=self.base_width),
            Bottleneck(2048,self.planeslist[3], groups=self.groups,base_width=self.base_width))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layers sequentially  
        x = self.layer1out(x)
        x = self.layer2out(x)
        x = self.layer3out(x)
        x = self.layer4out(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
def resnext50_32x4d():
    return ResNext(32 , 4)

class ResNext50(nn.Module):
    def __init__(self, init_weights=True):
        super(ResNext50, self).__init__()

        self.startlayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))
    
        self.bottleneckStartFirstlayer = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        self.bottleneckFirstlayer = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            

            nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        self.bottleneckStartSecondlayer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        self.bottleneckSecondlayer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            

            nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            

            nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        self.bottleneckStartThirdlayer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        self.bottleneckThirdlayer = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            
        self.bottleneckStartFourthlayer = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        

        self.bottleneckFourthlayer = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        self.downsample1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.downsample2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.downsample3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.downsample4 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        self.AvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.startlayer(x)
        identity = self.downsample1(x)
        x = self.bottleneckStartFirstlayer(x)
        x += identity
        x = self.bottleneckFirstlayer(x)
        x = self.relu(x)

        identity = self.downsample2(x)
        x = self.bottleneckStartSecondlayer(x)
        x += identity
        x = self.bottleneckSecondlayer(x)
        x = self.relu(x)

        identity = self.downsample3(x)
        x = self.bottleneckStartThirdlayer(x)
        x += identity
        x = self.bottleneckThirdlayer(x)
        x = self.relu(x)

        identity = self.downsample4(x)
        x = self.bottleneckStartFourthlayer(x)
        x += identity
        x = self.bottleneckFourthlayer(x)
        x = self.relu(x)

        x = self.AvgPool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  
