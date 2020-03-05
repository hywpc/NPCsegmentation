
"""
为图像数据增强提供一个通用模块
"""

import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import random
import torch


class RandomElasticDeformation(object):
    """
    对目标图像进行随机仿射变换,接受数据为3维或2维图像，数据类型为image, label, 维度均为ndarray(z,x,y)或ndarray(x,y)。
    """
    def __init__(self, image):
        assert isinstance(image, np.ndarray)

        random_state = np.random.RandomState(None)
        image_dimension = len(image.shape)
        if image_dimension == 2:
            shape = image.shape
        else:
            shape = image.shape[1:]
        alpha_affine = max(shape) * 0.08
        center_square = np.float32(shape) // 2
        square_size = min(shape) // 3

        self.image_dimension = image_dimension
        self.shape = shape
        self.pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size], center_square - square_size])
        self.pst2_plot = random_state.uniform(-alpha_affine, alpha_affine, size=self.pts1.shape).astype(np.float32)
        self.gaussian_filter_x = random_state.rand(*shape)
        self.gaussian_filter_y = random_state.rand(*shape)

    def __call__(self, image, label):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
             Convolutional Neural Networks applied to Visual Document Analysis", in
             Proc. of the International Conference on Document Analysis and
             Recognition, 2003.

        :param sample: 接受数据为3维或2维图像，数据类型为dict{'image': image, 'label': label}
                        image和label的维度均为ndarry(z,x,y)或ndarry(x,y)
        :return: 输出处理后的图像

         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        """

        alpha = max(self.shape) * 2
        sigma = max(self.shape) * 0.08

        if self.image_dimension == 2:
            # Random affine
            pts1 = self.pts1
            pts2 = pts1 + self.pst2_plot
            m = cv2.getAffineTransform(pts1, pts2)
            image = cv2.warpAffine(image, m, self.shape[::-1], borderMode=cv2.BORDER_REFLECT_101)
            label = cv2.warpAffine(label, m, self.shape[::-1], borderMode=cv2.BORDER_REFLECT_101)

            # elastic deformation
            dx = gaussian_filter((self.gaussian_filter_x * 2 - 1), sigma) * alpha
            dy = gaussian_filter((self.gaussian_filter_y * 2 - 1), sigma) * alpha
            x, y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            image = map_coordinates(image, indices, order=1, mode='reflect').reshape(self.shape)
            label = map_coordinates(label, indices, order=1, mode='reflect').reshape(self.shape)
            return image, label

        else:
            for i in range(image.shape[0]):
                # Random affine
                pts1 = self.pts1
                pts2 = pts1 + self.pst2_plot
                m = cv2.getAffineTransform(pts1, pts2)
                image[i, :, :] = cv2.warpAffine(image[i, :, :], m, self.shape[::-1], borderMode=cv2.BORDER_REFLECT_101)
                label[i, :, :] = cv2.warpAffine(label[i, :, :], m, self.shape[::-1], borderMode=cv2.BORDER_REFLECT_101)

                # elastic deformation
                dx = gaussian_filter((self.gaussian_filter_x * 2 - 1), sigma) * alpha
                dy = gaussian_filter((self.gaussian_filter_y * 2 - 1), sigma) * alpha
                x, y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
                indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
                image[i, :, :] = map_coordinates(image[i, :, :], indices, order=1, mode='reflect').reshape(self.shape)
                label[i, :, :] = map_coordinates(label[i, :, :], indices, order=1, mode='reflect').reshape(self.shape)
            return image, label


class RandomCrop(object):
    """
    对图片进行随机裁剪，裁剪为神经网络需要的维度


    Attributes:
        original_image:需要crop的原图
        output_image_shape: 输出图片尺寸
    """
    def __init__(self, original_image, output_image_shape):
        """
        根据原始图片尺寸和神经网络需要的图片尺寸初始化
        :param original_image: 原始图片，数据类别ndarray，
        :param output_image_shape: 输出图片尺寸，int或者tuple
        """
        assert isinstance(original_image, np.ndarray)
        assert isinstance(output_image_shape, (int, tuple))
        # 初始化原始图像的尺寸和维度
        self.original_image_shape = original_image.shape
        self.original_image_dimension = len(original_image.shape)
        # 初始化输出图像的尺寸需求
        if type(output_image_shape) == tuple:
            self.output_image_shape = output_image_shape
        else:
            self.output_image_shape = [output_image_shape for i in range(len(original_image.shape))]
        # 初始化随机crop的中点选点范围，随机数再每次crop时产生
        if self.original_image_dimension == 2:
            x, y = self.output_image_shape[0], self.output_image_shape[1]
            x_original, y_original = self.original_image_shape[0], self.original_image_shape[1]
            self.random_x, self.random_y = [0 + x // 2, x_original - x // 2], [0 + y // 2, y_original - y // 2]
        else:
            z, x, y = self.output_image_shape[0], self.output_image_shape[1], self.output_image_shape[2]
            z_original, x_original, y_original = self.original_image_shape[0], self.original_image_shape[1], self.original_image_shape[2]
            self.random_z, self.random_x, self.random_y = [0 + z // 2, z_original - z // 2], [0 + x // 2, x_original - x // 2], [0 + y // 2, y_original - y // 2]

    def __call__(self, image, label):
        if self.original_image_dimension == 2:
            random_x, random_y = random.randint(self.random_x[0], self.random_x[1]), random.randint(self.random_y[0], self.random_y[1])
            crop_pixel_x_y = self.output_image_shape[0] // 2
            image = image[random_x - crop_pixel_x_y: random_x + crop_pixel_x_y, random_y - crop_pixel_x_y: random_y + crop_pixel_x_y]
            label = label[random_x - crop_pixel_x_y: random_x + crop_pixel_x_y, random_y - crop_pixel_x_y: random_y + crop_pixel_x_y]
            return image, label
        else:
            random_z, random_x, random_y = random.randint(self.random_z[0], self.random_z[1]), random.randint(self.random_x[0], self.random_x[1]), random.randint(self.random_y[0], self.random_y[1])
            crop_pixel_z, crop_pixel_x_y = self.output_image_shape[0] // 2, self.output_image_shape[1] // 2
            image = image[random_z - crop_pixel_z: random_z + crop_pixel_z, random_x - crop_pixel_x_y: random_x + crop_pixel_x_y, random_y - crop_pixel_x_y: random_y + crop_pixel_x_y]
            label = label[random_z - crop_pixel_z: random_z + crop_pixel_z, random_x - crop_pixel_x_y: random_x + crop_pixel_x_y, random_y - crop_pixel_x_y: random_y + crop_pixel_x_y]
            return image, label


class ToTensor(object):
    """
    将目标image和label从ndarrys转为torch.tensor
    """

    def __call__(self, sample):
        sample = torch.from_numpy(sample).unsqueeze(0)
        return sample


class LabelScatter(object):
    """对多分类任务的label进行独热处理，方便某些特殊的loss函数使用

    独热处理

    Attributes:
        label_class_number: 分为几类

    """
    def __init__(self, label_class_number):
        self.label_class_number = label_class_number

    def __call__(self, data):
        label_class_number = self.label_class_number
        data = torch.zeros(label_class_number, data.shape[1], data.shape[2], data.shape[3]).scatter_(0, data.long(), 1)
        return data


class RandomScale(object):
    """
    对图片进行随机缩放
    """
    def __init__(self, image):
        pass