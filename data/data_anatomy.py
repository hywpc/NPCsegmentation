# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import SimpleITK as sitk
from data.DataAugmentation import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()  # interactive mode


class NPCDataSet(Dataset):
    """NPC MRI dataset"""

    def __init__(self, file, scatter_label=False, transform=True, crop=True, crop_size=(80, 160, 160), mode='train'):
        """
        Args:
            file (string): path to the landmarks file with annotation,
        """
        super().__init__()
        img_num = os.listdir(file)
        img_path = []
        lbl_path = []

        # 获取每个nii文件详细地址，存入list中
        for i in img_num:
            img = os.path.join(file, i, 'data.nii.gz')
            lbl = os.path.join(file, i, 'label.nii.gz')

            img_path.append(img)
            lbl_path.append(lbl)

        self.img_path = img_path
        self.lbl_path = lbl_path
        self.scatter_label = scatter_label
        self.transform = transform
        self.crop = crop
        self.crop_size = crop_size
        self.mode = mode

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_dir = self.img_path[idx]
        lbl_dir = self.lbl_path[idx]
        scatter_label = self.scatter_label
        transform = self.transform
        crop = self.crop
        crop_size = self.crop_size
        mode = self.mode

        # 根据img_dir读取具体照片
        image = sitk.ReadImage(img_dir)
        image = sitk.GetArrayFromImage(image)

        z = image.shape[0]
        x = 245
        y = 256

        # 选取范围, 可以使用tensor.clamp(-200, 700)
        image[image > 700] = 700
        image[image < -200] = -200
        # 归一化
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        label = sitk.ReadImage(lbl_dir)
        label = sitk.GetArrayFromImage(label)
        if mode == 'test':
            image = image[z-90: z-10, x - 80: x + 80, y - 80: y + 80]
            label = label[z-90: z-10, x - 80: x + 80, y - 80: y + 80]
        else:
            image = image[:, x - 128: x + 128, y - 128: y + 128]
            label = label[:, x - 128: x + 128, y - 128: y + 128]

        if transform:
            elastic_deformation = RandomElasticDeformation(image)
            image, label = elastic_deformation(image, label)
        if crop:
            random_crop = RandomCrop(image, crop_size)
            image, label = random_crop(image, label)

        to_tensor = ToTensor()
        image = to_tensor(image)
        label = to_tensor(label)

        # 根据loss需求是否对label独热处理
        if scatter_label:
            label_scatter = LabelScatter(23)
            label = label_scatter(label)

        sample = {'image': image, 'label': label}
        return sample


if __name__ == '__main__':
    # a = NPCDataSet('/home/hyw/data/HaN_OAR/train/')
    a = NPCDataSet('E:\\3.Work\\data\\HaN_OAR\\train', scatter_label=True, transform=False, crop=False, crop_size=(80, 160, 160), mode='test')
    print(a[7]['image'].shape)
    for i in a:
        x = i['image'].shape
        y = i['label'].shape
        print('image size:', x, 'label size',y)
