from net.Net_1pool import *
from data.data_anatomy import *
import SimpleITK as sitk
import numpy as np
from metric.metric import DiceCoefficient
import torch


# 完成预测
def evaluate_prediction(data, metric):
    for i, data in enumerate(data):
        input_data = data['image'].float().unsqueeze(0).cuda().requires_grad_(False)
        label = data['label'].unsqueeze(0).cuda().requires_grad_(False)
        with torch.no_grad():
            predict = net(input_data)
        result = metric(predict, label).cpu().numpy()
        # print(result)
        # print(type(result))
        x = [j for j in result]
        for k in range(len(x)):
            x[k] = float('%.4f' % x[k])
        print(len(x), x)
        print(float('%.4f' % np.mean(x)))


def save_prediction(test_data, dir):
    with torch.no_grad():
        for i, data in enumerate(test_data):
            image = data['image'].squeeze()
            image1 = sitk.GetImageFromArray(image)
            sitk.WriteImage(image1, '{}\\test_image{}.nii.gz'.format(dir, i + 1))

            input_data = data['image'].float().unsqueeze(0).cuda()
            label = data['label']

            predict = net(input_data).cpu().numpy().squeeze().argmax(0).astype(np.int16)
            print(label.shape)
            label = label.squeeze().numpy().astype(np.int16)

            predict = sitk.GetImageFromArray(predict)
            sitk.WriteImage(predict, '{}\\test_prediction{}.nii.gz'.format(dir, i + 1))

            label = sitk.GetImageFromArray(label)
            sitk.WriteImage(label, '{}\\test_label{}.nii.gz'.format(dir, i + 1))

if __name__ == '__main__':
    # 初始化网络
    net = AnatomyNet(classes=23)
    # net.load_state_dict(torch.load("/home/hyw/Head_Neck_Seg/parameters/with_deep_supervision/net_params465.pkl"))
    net.load_state_dict(torch.load("E:/3.Work/siat_project/params/net_params998.pkl"))
    net = net.cuda()
    # 初始化测试系数
    test = DiceCoefficient(ignore_index=None, classes_mean=False)
    # 初始化测试集数据集
    test_data_set = NPCDataSet('E:/3.Work/data/HaN_OAR/test', scatter_label=True, transform=False, crop=False, crop_size=(80, 160, 160), mode='test')

    #save_prediction(test_data_set, dir='test_result')
    evaluate_prediction(test_data_set, metric=test)
