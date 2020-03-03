# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 20:07:26 2019

@author: hyw
"""

import numpy as np
import nrrd
import os
from torch.utils.data import Dataset
import random
import torch

class H_N_data(Dataset):
    def __init__(self, file):
        img_path = os.listdir(file)
        label_path = '{}/{}/structures'.format(file, img_path[0])
        lbl_path = os.listdir(label_path)
        print(img_path)
        self.file = file
        self.img_path = img_path
        self.label_path = lbl_path
        
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        
        img_num = self.img_path[idx]
        img_dir = os.path.join(self.file, img_num, 'img.nrrd')
        image, image_options = nrrd.read(img_dir)

        z = image.shape[2]
        x = random.randint(200, 300)
        y = random.randint(200, 300)
        
        label = np.zeros(image.shape)
        for j in range(len(self.label_path)):
            label_dir = os.path.join(self.file, img_num, 'structures', self.label_path[j])
            nrrd_data, nrrd_options = nrrd.read(label_dir)
            a = nrrd_data > 0
            nrrd_data[a] = j+1
            label += nrrd_data
            
        label_dir = os.path.join(self.file, img_num, 'structures', 'Mandible.nrrd')
        nrrd_data, nrrd_options = nrrd.read(label_dir)
        label = nrrd_data
        image[image>200]=200
        image[image<-200]=-200
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = image.transpose((2,1,0))
        image = image[z-80: z+1, x-128:x+128, y-128:y+128]
        image = torch.from_numpy(image) 
        image = image.type(torch.FloatTensor).unsqueeze(0)
        
        label = label.transpose((2,1,0))
        label = label[z-80: z+1, x-128:x+128, y-128:y+128]
        #print(img_num,np.max(label))
        label = torch.LongTensor(label).unsqueeze(0)
        #label = torch.zeros(10,label.shape[1],label.shape[2],label.shape[3]).scatter_(0, label, 1)
        #label = label.type(torch.LongTensor)
        
        sample = {'image':image, 'label': label}
        
        return sample


if __name__ == '__main__':
    a = H_N_data('/home/hyw/Head_Neck_Seg/HeadandNeck/train/')
    for i in range(len(a)):
        print(a[i]['label'].shape)
        print(a[i]['image'].shape)