# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import print_function, division
import os
import torch
import random


import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader



import SimpleITK as sitk

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class NPC_dataset(Dataset):
    """NPC MRI dataset"""
    
    def __init__(self, file, transform_image = None, transform_landmarks = None):
        """
        Args:
            filr (string): path to the landmarks file with annotation, 
                must be csv.
            root_dir(string): Directort of all images.
            transform (callable, optional): optional transform 
                to be applied on a sample.
        """
        img_num = os.listdir(file)
        img_path = []
        lbl_path = []

            
        for i in img_num:
            img = os.path.join(file, i, 'data.nii.gz') #获取每个nii文件地址
            lbl = os.path.join(file, i, 'label.nii.gz')
            
            img_path.append(img)
            lbl_path.append(lbl)

            #img = nib.load(img) #使用nibabel快速读取单个nii文件的图片数
            #lbl = nib.load(lbl)
            #for j in range(img.shape[2]):
                #img_path.append((i,j)) # (x, y) 第x个患者的第y张图片
                #lbl_path.append((i,j))
        
        self.file = file
        self.img_path = img_path
        self.lbl_path = lbl_path
        
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        
        img_dir = self.img_path[idx]
        lbl_dir = self.lbl_path[idx]
        # 根据img_dir定位图组

        # 根据img_dir读取具体照片
        image = sitk.ReadImage(img_dir)
        image = sitk.GetArrayFromImage(image)
                
        z = image.shape[0]
        x = 245
        y = 256
        
        image = image[z-80: z-16, x-80:x+80, y-80:y+80]
        
        landmarks = sitk.ReadImage(lbl_dir)
        landmarks = sitk.GetArrayFromImage(landmarks)
        landmarks = landmarks[z-80: z-16, x-80:x+80, y-80:y+80]
        
        #图像归一化
        image[image>200]=200
        image[image<-200]=-200
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        image = image[np.newaxis, :, :, :]
        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        
        landmarks = landmarks[np.newaxis, :, :, :]
        #landmarks = torch.from_numpy(landmarks)
        landmarks = torch.LongTensor(landmarks)#.unsqueeze(0)

        landmarks = torch.zeros(23,landmarks.shape[1],landmarks.shape[2],landmarks.shape[3]).scatter_(0, landmarks, 1)
        
        
        sample = {'image':image, 'label': landmarks}
        
        return sample
    
if __name__ == '__main__':
    a = NPC_dataset('/home/hyw/Head_Neck_Seg/HaN_OAR/train/')
    for i in range(len(a)):
        print(a[i]['label'].shape)
        print(a[i]['image'].shape)