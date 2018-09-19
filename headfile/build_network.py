
# coding: utf-8

# In[3]:


import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F


# simple 4-layer FCN
class FULLY_CONV_NET(nn.Module):
    def __init__(self,num_input_channel):
        super(FULLY_CONV_NET, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_input_channel, 100,kernel_size=5, stride=1,padding=2),  
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 50,kernel_size=7, stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(50, 10,kernel_size=7, stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 1,kernel_size=3, stride=1,padding=1),
        )

    def forward(self, x):
        out = self.features(x)
        return out



# simple 4-layer FCN
class TEST_NET_1(nn.Module):
    def __init__(self,num_input_channel):
        super(TEST_NET_1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_input_channel, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=1),
        )

    def forward(self, x):
        out = self.features(x)
        return out

    

