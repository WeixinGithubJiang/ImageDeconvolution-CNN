
# coding: utf-8

# In[1]:


import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# In[2]:


def load_dataset(path,filename):
    f = h5py.File(path+filename)
    input_dataset = np.asarray([f[element[0]][:] for element in f['input_dataset']],dtype=np.float32)
    output_dataset = np.asarray([f[element[0]][:] for element in f['output_dataset']],dtype=np.float32)
    f.close()
    return input_dataset,output_dataset


# In[18]:


class DeconvolutionDataset(Dataset):
    """dataset."""

    def __init__(self, input_dataset,output_dataset, transform=None):
        """

        """
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset
        self.transform = transform

    def __len__(self):
        return len(self.output_dataset)

    def __getitem__(self, idx):

        sample = {'input': self.input_dataset[idx,:,:,:], 'output': np.reshape(self.output_dataset[idx,:,:],(1,128,128))}
        
        if self.transform:
            sample = self.transform(sample)

        return sample


# In[4]:


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input_image, output_image = sample['input'], sample['output']
        return {'input': torch.from_numpy(input_image),
                'output': torch.from_numpy(output_image)}
    
class ImageSelect(object):
    """select the first n images"""
    def __init__(self, select_num):
        assert isinstance(select_num, int)
        assert select_num < 201
        assert select_num > 0
        self.select_num = select_num

    def __call__(self, sample):
        input_image, output_image = sample['input'], sample['output']

#         select_image = np.copy(input_image[:self.select_num])

        return {'input': np.copy(input_image[:self.select_num,:,:]), 'output': output_image}
    

