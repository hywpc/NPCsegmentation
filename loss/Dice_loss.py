"""

Dice loss
用来处理分割过程中的前景背景像素非平衡的问题
"""

import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, prediction, target):
        N = target.size(0)
        smooth = 1

        prediction_flat = prediction.view(N, -1)
        target_flat = target.view(N, -1)
        
        intersection = prediction_flat * target_flat

        loss = (2 * intersection.sum() + smooth) / (prediction_flat.sum() + target_flat.sum() + smooth)
        # print(loss, type(loss))
        loss = 1 - loss / N

        return loss


class MultiClassDiceLoss(nn.Module):
    """
    需要独热处理，分别对每一类做DiceLoss。
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
    batch size and C is number of classes
    """
    def __init__(self):
        super(MultiClassDiceLoss, self).__init__()
 
    def forward(self, prediction, target, weights=None):
        
        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes
        dice = DiceLoss()
        totalLoss = 0
        
        for i in range(C):
            diceLoss = dice(prediction[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
                
            totalLoss += diceLoss

        return totalLoss
