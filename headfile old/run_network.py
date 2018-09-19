
# coding: utf-8

# In[ ]:


from skimage.measure import compare_ssim as ssim
import csv
import cv2 as cv
import os

# In[ ]:


def train(train_loader,model,criterion, optimizer, num_epochs,device):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, sample in enumerate(train_loader):
            inputs,target = sample['input'].to(device),sample['output'].to(device)

            # forward
            out = model(inputs)
            loss = criterion(out, target)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch < 100:
            print('Epoch[{}/{}], loss: {:.6f}'
                  .format(epoch+1, num_epochs, loss.data[0]))
        elif (epoch+1) % 100 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'
                  .format(epoch+1, num_epochs, loss.data[0]))


# In[ ]:


def test(test_data,model,device):
    model.eval()
    predict = model(test_data.to(device))
    predict_test = predict.data.cpu().numpy()
    return predict_test


# In[ ]:


"""
Video Quality Metrics
Copyright (c) 2014 Alex Izvorski <aizvorski@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy
import math

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# In[ ]:


def normalize_image(img):
    return img/(img.max()-img.min())*255


def save_result(save_path,input_num,psnr_value,ssim_value,predict_test,test_output_dataset,save_filename):
    filepath = os.path.join(save_path,input_num)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    for i in range(4):
        filename = 'test_predict_'+str(i)+'.png'
        cv.imwrite(os.path.join(filepath,filename),normalize_image(predict_test[i,0,:,:]))
        psnr_value.append('%.4f' % psnr(normalize_image(predict_test[i,0,:,:]),normalize_image(test_output_dataset[i,:,:])))
        ssim_value.append('%.4f' % ssim(normalize_image(predict_test[i,0,:,:]), normalize_image(test_output_dataset[i,:,:]), data_range=255))
    metric_value = [psnr_value,ssim_value]
    if not os.path.isfile(os.path.join(save_path,save_filename)):
        myfile = open(os.path.join(save_path,save_filename),'w')
    else:
        myfile = open(os.path.join(save_path,save_filename),'a')
    with myfile:
        writer = csv.writer(myfile)
        writer.writerows(metric_value)
    myfile.close()
        
