# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import print_function, division
import os
import torch
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

        # 获取每个nii文件详细地址，存入list中
        for i in img_num:
            img = os.path.join(file, i, 'data.nii.gz')
            lbl = os.path.join(file, i, 'label.nii.gz')
            
            img_path.append(img)
            lbl_path.append(lbl)
        
        self.file = file
        self.img_path = img_path
        self.lbl_path = lbl_path
        self.Loss = Loss
        
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        sample = {}
        img_dir = self.img_path[idx]
        lbl_dir = self.lbl_path[idx]
        Loss = self.Loss
        # 根据img_dir定位图组

        # 根据img_dir读取具体照片
        image = sitk.ReadImage(img_dir)
        image = sitk.GetArrayFromImage(image)
        image[image>600]=600
        image[image<-200]=-200
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        label = sitk.ReadImage(lbl_dir)
        label = sitk.GetArrayFromImage(label)

        label_is_i = label == 11
        label_not_i = label != 11
        label_test = label.copy()
        label_test[label_is_i] = 1
        label_test[label_not_i] = 0

        item_index = np.argwhere(label_test == 1)
        z = (np.min(item_index, 0)[0] + np.max(item_index, 0)[0]) // 2 - 15
        x = (np.min(item_index, 0)[1] + np.max(item_index, 0)[1]) // 2
        y = (np.min(item_index, 0)[2] + np.max(item_index, 0)[2]) // 2
        image_out = image[z - 32:z + 32, x - 32:x + 32, y - 32:y + 32]
        label_out = label_test[z - 32:z + 32, x - 32:x + 32, y - 32:y + 32]

        image_out = torch.FloatTensor(image_out).unsqueeze(0)
        label_out = torch.LongTensor(label_out).unsqueeze(0)

        if self.Loss == 'Dice':
            label_out = torch.zeros(2, label_out.shape[1], label_out.shape[2], label_out.shape[3]).scatter_(0, label_out, 1)


        sample = {'image_out': image_out, 'label_out': label_out}
        return sample
    
if __name__ == '__main__':
    #a = NPC_dataset('/home/hyw/Head_Neck_Seg/HaN_OAR/train/')
    a = NPC_dataset('E:\\3.Work\\data\\HaN_OAR\\train')#, Loss= 'Dice')
    for i in range(40):
        print(i)
        x = a[i]
        for i in x:
            print(x[i].shape)