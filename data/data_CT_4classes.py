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
    
    def __init__(self, file, transform_image = None, transform_landmarks = None, Loss = None):
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
        self.Loss = Loss
        
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        
        img_dir = self.img_path[idx]
        lbl_dir = self.lbl_path[idx]
        Loss = self.Loss
        # 根据img_dir定位图组

        # 根据img_dir读取具体照片
        image = sitk.ReadImage(img_dir)
        image = sitk.GetArrayFromImage(image)
                
        z = image.shape[0]
        x = 245
        y = 256
        
        image[image>600]=600
        image[image<-200]=-200
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        
        image1 = image[z-80: z-16, x-96:x, y-96:y]
        image2 = image[z-80: z-16, x:x+96, y-96:y]
        image3 = image[z-80: z-16, x-96:x, y:y+96]
        image4 = image[z-80: z-16, x:x+96, y:y+96]
        image = image[z-80: z-16, x-96:x+96, y-96:y+96]
        
        label = sitk.ReadImage(lbl_dir)
        label = sitk.GetArrayFromImage(label)
        
        
        label1 = label[z-80: z-16, x-96:x, y-96:y]
        label2 = label[z-80: z-16, x:x+96, y-96:y]
        label3 = label[z-80: z-16, x-96:x, y:y+96]
        label4 = label[z-80: z-16, x:x+96, y:y+96]
        label = label[z-80: z-16, x-96:x+96, y-96:y+96]
        
        if not True:
            b = np.zeros(label.shape,np.int)
            a = label == 1
            b[a] = 1
            a = label == 9
            b[a] = 2
            a = label == 10
            b[a] = 2
            a = label == 22
            b[a] = 3
            a = label == 21
            b[a] = 3
            a = label == 12
            b[a] = 4
            a = label == 13
            b[a] = 4

        image = torch.FloatTensor(image).unsqueeze(0)
        image1 = torch.FloatTensor(image1).unsqueeze(0)
        image2 = torch.FloatTensor(image2).unsqueeze(0)
        image3 = torch.FloatTensor(image3).unsqueeze(0)
        image4 = torch.FloatTensor(image4).unsqueeze(0)
        
        #label = torch.from_numpy(label).unsqueeze(0)
        label = torch.LongTensor(label).unsqueeze(0)
        label1 = torch.LongTensor(label1).unsqueeze(0)
        label2 = torch.LongTensor(label2).unsqueeze(0)
        label3 = torch.LongTensor(label3).unsqueeze(0)
        label4 = torch.LongTensor(label4).unsqueeze(0)

        if self.Loss == 'Dice':
            label = torch.zeros(23,label.shape[1],label.shape[2],label.shape[3]).scatter_(0, label, 1)
            label1 = torch.zeros(23,label1.shape[1],label1.shape[2],label1.shape[3]).scatter_(0, label1, 1)
            label2 = torch.zeros(23,label2.shape[1],label2.shape[2],label2.shape[3]).scatter_(0, label2, 1)
            label3 = torch.zeros(23,label3.shape[1],label3.shape[2],label3.shape[3]).scatter_(0, label3, 1)
            label4 = torch.zeros(23,label4.shape[1],label4.shape[2],label4.shape[3]).scatter_(0, label4, 1)
        
        
        sample = {'image':image, 'image1':image1, 'image2':image2, 'image3':image3, 'image4':image4, \
                  'label': label, 'label1':label1, 'label2':label2, 'label3':label3, 'label4':label4}
        
        return sample
    
if __name__ == '__main__':
    #a = NPC_dataset('/home/hyw/Head_Neck_Seg/HaN_OAR/train/')
    a = NPC_dataset('E:\\3.Work\\data\\HaN_OAR\\train')#, Loss = 'Dice')
    x = a[4]
    for i in x:
        print(x[i].shape)