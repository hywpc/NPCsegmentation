import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, prediction, true):
        prediction = prediction.clamp(min=1e-5, max=1.0)
        b = prediction.size(0)
        p = prediction.view(b, -1)
        t = true.view(b, -1)
        coff = 2 * (p * t).sum(1) / (p.sum(1) + t.sum(1))
        loss = 1 - coff.mean(0)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, power=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.power = power
        self.alpha = alpha

    def forward(self, prediction, true):
        prediction = prediction.clamp(min=1e-5, max=1.0)
        b, c, w, h, d = prediction.size()
        p = prediction.view(b, -1)
        t = true.view(b, -1)
        loss = self.alpha * t * (1 - p).pow(self.power) * p.log()
        loss = loss.sum(1) / (w * h * d)
        loss = -loss.mean(0)
        return loss


class HybridLoss(nn.Module):
    def __init__(self, lambda_a=0.5, classes=23):
        super(HybridLoss, self).__init__()
        self.dice_loss1 = DiceLoss()
        self.focal_loss2 = FocalLoss()
        self.lambda_a = lambda_a
        self.classes = classes

    def forward(self, prediction, true):
        dice_loss11 = self.dice_loss1
        focal_loss22 = self.focal_loss2
        lambda_a = self.lambda_a
        classes = self.classes

        loss1 = dice_loss11(prediction, true)
        loss2 = focal_loss22(prediction, true)
        total_loss = loss1 + lambda_a * loss2
        total_loss = total_loss * classes
        return total_loss


if __name__ == '__main__':
    b = FocalLoss()
    print(b.alpha)
    a = HybridLoss(lambda_a=0.6, classes=23)
    print(a.lambda_a)