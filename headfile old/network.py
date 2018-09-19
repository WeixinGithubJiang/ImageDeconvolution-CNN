
# coding: utf-8

# In[3]:


import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F


# In[6]:


# simple 4-layer FCN
class FullyConvNet_1(nn.Module):
    def __init__(self):
        super(FullyConvNet_1, self).__init__()
        self.conv1 = nn.Conv2d(200, 100,kernel_size=5, stride=1,padding=2)  
        self.conv2 = nn.Conv2d(100, 50,kernel_size=7, stride=1,padding=3)
        self.conv3 = nn.Conv2d(50, 10,kernel_size=7, stride=1,padding=3)
        self.conv4 = nn.Conv2d(10, 1,kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_layer1 = self.relu(self.conv1(x))
        out_layer2 = self.relu(self.conv2(out_layer1))
        out_layer3 = self.relu(self.conv3(out_layer2))
        out = self.conv4(out_layer3)
        return out

# simple 4-layer FCN
class FullyConvNet_200(nn.Module):
    def __init__(self):
        super(FullyConvNet_200, self).__init__()
        self.conv1 = nn.Conv2d(200, 100,kernel_size=5, stride=1,padding=2)  
        self.conv2 = nn.Conv2d(100, 50,kernel_size=7, stride=1,padding=3)
        self.conv3 = nn.Conv2d(50, 10,kernel_size=7, stride=1,padding=3)
        self.conv4 = nn.Conv2d(10, 1,kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_layer1 = self.relu(self.conv1(x))
        out_layer2 = self.relu(self.conv2(out_layer1))
        out_layer3 = self.relu(self.conv3(out_layer2))
        out = self.conv4(out_layer3)
        return out   
    
# simple 4-layer FCN
class FullyConvNet_175(nn.Module):
    def __init__(self):
        super(FullyConvNet_175, self).__init__()
        self.conv1 = nn.Conv2d(175, 100,kernel_size=5, stride=1,padding=2)  
        self.conv2 = nn.Conv2d(100, 50,kernel_size=7, stride=1,padding=3)
        self.conv3 = nn.Conv2d(50, 10,kernel_size=7, stride=1,padding=3)
        self.conv4 = nn.Conv2d(10, 1,kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_layer1 = self.relu(self.conv1(x))
        out_layer2 = self.relu(self.conv2(out_layer1))
        out_layer3 = self.relu(self.conv3(out_layer2))
        out = self.conv4(out_layer3)
        return out    

# simple 4-layer FCN
class FullyConvNet_150(nn.Module):
    def __init__(self):
        super(FullyConvNet_150, self).__init__()
        self.conv1 = nn.Conv2d(150, 100,kernel_size=5, stride=1,padding=2)  
        self.conv2 = nn.Conv2d(100, 50,kernel_size=7, stride=1,padding=3)
        self.conv3 = nn.Conv2d(50, 10,kernel_size=7, stride=1,padding=3)
        self.conv4 = nn.Conv2d(10, 1,kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_layer1 = self.relu(self.conv1(x))
        out_layer2 = self.relu(self.conv2(out_layer1))
        out_layer3 = self.relu(self.conv3(out_layer2))
        out = self.conv4(out_layer3)
        return out  

# simple 4-layer FCN
class FullyConvNet_125(nn.Module):
    def __init__(self):
        super(FullyConvNet_125, self).__init__()
        self.conv1 = nn.Conv2d(125, 100,kernel_size=5, stride=1,padding=2)  
        self.conv2 = nn.Conv2d(100, 50,kernel_size=7, stride=1,padding=3)
        self.conv3 = nn.Conv2d(50, 10,kernel_size=7, stride=1,padding=3)
        self.conv4 = nn.Conv2d(10, 1,kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_layer1 = self.relu(self.conv1(x))
        out_layer2 = self.relu(self.conv2(out_layer1))
        out_layer3 = self.relu(self.conv3(out_layer2))
        out = self.conv4(out_layer3)
        return out    
    
# simple 4-layer FCN
class FullyConvNet_100(nn.Module):
    def __init__(self):
        super(FullyConvNet_100, self).__init__()
        self.conv1 = nn.Conv2d(100, 100,kernel_size=5, stride=1,padding=2)  
        self.conv2 = nn.Conv2d(100, 50,kernel_size=7, stride=1,padding=3)
        self.conv3 = nn.Conv2d(50, 10,kernel_size=7, stride=1,padding=3)
        self.conv4 = nn.Conv2d(10, 1,kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_layer1 = self.relu(self.conv1(x))
        out_layer2 = self.relu(self.conv2(out_layer1))
        out_layer3 = self.relu(self.conv3(out_layer2))
        out = self.conv4(out_layer3)
        return out

# simple 4-layer FCN
class FullyConvNet_75(nn.Module):
    def __init__(self):
        super(FullyConvNet_75, self).__init__()
        self.conv1 = nn.Conv2d(75, 100,kernel_size=5, stride=1,padding=2)  
        self.conv2 = nn.Conv2d(100, 50,kernel_size=7, stride=1,padding=3)
        self.conv3 = nn.Conv2d(50, 10,kernel_size=7, stride=1,padding=3)
        self.conv4 = nn.Conv2d(10, 1,kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_layer1 = self.relu(self.conv1(x))
        out_layer2 = self.relu(self.conv2(out_layer1))
        out_layer3 = self.relu(self.conv3(out_layer2))
        out = self.conv4(out_layer3)
        return out    
    
# simple 4-layer FCN
class FullyConvNet_50(nn.Module):
    def __init__(self):
        super(FullyConvNet_50, self).__init__()
        self.conv1 = nn.Conv2d(50, 100,kernel_size=5, stride=1,padding=2)  
        self.conv2 = nn.Conv2d(100, 50,kernel_size=7, stride=1,padding=3)
        self.conv3 = nn.Conv2d(50, 10,kernel_size=7, stride=1,padding=3)
        self.conv4 = nn.Conv2d(10, 1,kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_layer1 = self.relu(self.conv1(x))
        out_layer2 = self.relu(self.conv2(out_layer1))
        out_layer3 = self.relu(self.conv3(out_layer2))
        out = self.conv4(out_layer3)
        return out
    
# simple 4-layer FCN
class FullyConvNet_25(nn.Module):
    def __init__(self):
        super(FullyConvNet_25, self).__init__()
        self.conv1 = nn.Conv2d(25, 100,kernel_size=5, stride=1,padding=2)  
        self.conv2 = nn.Conv2d(100, 50,kernel_size=7, stride=1,padding=3)
        self.conv3 = nn.Conv2d(50, 10,kernel_size=7, stride=1,padding=3)
        self.conv4 = nn.Conv2d(10, 1,kernel_size=3, stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_layer1 = self.relu(self.conv1(x))
        out_layer2 = self.relu(self.conv2(out_layer1))
        out_layer3 = self.relu(self.conv3(out_layer2))
        out = self.conv4(out_layer3)
        return out

    
class TestNet(nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(200, 32, kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7,padding=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5,padding=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3,padding=1)
        self.conv6 = nn.Conv2d(512, 256, kernel_size=3,padding=1)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=3,padding=1)
        self.conv8 = nn.Conv2d(128, 64, kernel_size=5,padding=2)
        self.conv9 = nn.Conv2d(64, 32, kernel_size=5,padding=2)
        self.conv10 = nn.Conv2d(32, 16, kernel_size=7,padding=3)
        self.conv11 = nn.Conv2d(16, 8, kernel_size=9,padding=4)
        self.conv12 = nn.Conv2d(8, 4, kernel_size=9,padding=4)
        self.conv13 = nn.Conv2d(4, 2, kernel_size=9,padding=4)
        self.conv14 = nn.Conv2d(2, 1, kernel_size=9,padding=4)


    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
        x6 = F.relu(self.conv6(x5))+x4
        x7 = F.relu(self.conv7(x6))+x3
        x8 = F.relu(self.conv8(x7))+x2
        x9 = F.relu(self.conv9(x8))+x1
        x10 = F.relu(self.conv10(x9))
        x11 = F.relu(self.conv11(x10))
        x12 = F.relu(self.conv12(x11))
        x13 = F.relu(self.conv13(x12))
        x = self.conv14(x13)
        return x

# In[7]:


# encoding-decoding FCN
class FullyConvNet_2(nn.Module):
    def __init__(self):
        super(FullyConvNet_2, self).__init__()
        self.conv1 = nn.Conv2d(200, 100,kernel_size=3, stride=1,padding=1)  
        self.conv1_bn = nn.BatchNorm2d(100)
        self.conv_tmp1 = nn.Conv2d(100, 50,kernel_size=3, stride=1,padding=1)  
        self.conv_tmp1_bn = nn.BatchNorm2d(50)
        self.conv_tmp2 = nn.Conv2d(50, 25,kernel_size=3, stride=1,padding=1)
        self.conv_tmp2_bn = nn.BatchNorm2d(25)
        self.conv_tmp3 = nn.Conv2d(25, 16,kernel_size=1, stride=1,padding=0)
        self.conv_tmp3_bn = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32,kernel_size=5, stride=1,padding=2)
        self.conv2_bn = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 64,kernel_size=7, stride=1,padding=3)
#         self.conv3_bn = nn.BatchNorm2d(64)
#         self.conv4 = nn.Conv2d(64, 128,kernel_size=3, stride=1,padding=1)
#         self.conv4_bn = nn.BatchNorm2d(128)
        
#         self.conv5 = nn.Conv2d(128, 64,kernel_size=3, stride=1,padding=1)
#         self.conv5_bn = nn.BatchNorm2d(64)
#         self.conv6 = nn.Conv2d(64, 32,kernel_size=7, stride=1,padding=3)
#         self.conv6_bn = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(32, 16,kernel_size=5, stride=1,padding=2)
        self.conv8 = nn.Conv2d(16, 1,kernel_size=1, stride=1,padding=0)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out_layer1 = self.relu(self.conv1(x))
        
        out_layer_tmp1 = self.relu(self.conv_tmp1(self.conv1_bn(out_layer1)))
        out_layer_tmp2 = self.relu(self.conv_tmp2(self.conv_tmp1_bn(out_layer_tmp1)))
        out_layer_tmp3 = self.relu(self.conv_tmp3(self.conv_tmp2_bn(out_layer_tmp2)))
        
        out_layer2 = self.relu(F.max_pool2d(self.conv2(self.conv_tmp3_bn(out_layer_tmp3)),2))
#         out_layer3 = self.relu(F.max_pool2d(self.conv3(self.conv2_bn(out_layer2)),2))
#         out_layer4 = self.relu(F.max_pool2d(self.conv4(self.conv3_bn(out_layer3)),2))
        
#         out_layer5 = self.relu(self.conv5(F.upsample(self.conv4_bn(out_layer4),scale_factor=2)))
#         out_layer6 = self.relu(self.conv6(F.upsample(self.conv5_bn(out_layer5),scale_factor=2)))
#         out_layer7 = self.relu(self.conv7(F.upsample(self.conv6_bn(out_layer6),scale_factor=2)))
        out_layer7 = self.relu(self.conv7(F.upsample(self.conv2_bn(out_layer2),scale_factor=2)))
        
        out = self.conv8(out_layer7)
        return out

