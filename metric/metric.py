import torch


class DiceCoefficient():
    """计算dice系数，可用于多分类的情况
    https://arxiv.org/pdf/1707.03237.pdf
    Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """
    def __init__(self, epsilon=1e-5, ignore_index=None, classes_mean=True):
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.classes_mean = classes_mean

    @staticmethod
    def flatten(tensor):
        """将tensor的shape由(N, C, D, H, W) -> (C, N * D * H * W)
        :param tensor: torch.tensor
        :return: 按channel抹平的tensor
        """
        if len(tensor.size()) == 5:
            c = tensor.size(1)
            # new axis order
            axis_order = (1, 0) + tuple(range(2, tensor.dim()))
            # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
            transposed = tensor.permute(axis_order)
            # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
            return transposed.reshape(c, -1)

        elif len(tensor.size()) == 4:
            c = tensor.size(0)
            return tensor.reshape(c, -1)

    @staticmethod
    def compute_per_channel_dice(prediction, target, epsilon=1e-5):

        prediction = DiceCoefficient.flatten(prediction)
        target = DiceCoefficient.flatten(target)

        target = target.float()
        # Compute per channel Dice Coefficient
        intersect = (prediction * target).sum(-1)
        if weight is not None:
            intersect = weight * intersect

        denominator = (prediction + target).sum(-1)
        return 2.0 * intersect / denominator.clamp(min=epsilon)

    def __call__(self, prediction, target):
        """
        :param prediction: 5D probability maps torch tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: Soft Dice Coefficient averaged over all channels/classes
        """
        # Average across channels in order to get the final score
        classes_mean = self.classes_mean
        if classes_mean:
            return torch.mean(self.compute_per_channel_dice(prediction, target)
        else:
            return self.compute_per_channel_dice(prediction, target)


if __name__ == '__main__':
    test = DiceCoefficient(classes_mean=True)
    fake_data = torch.randn((1, 23, 80, 160, 160), requires_grad=False)
    fake_label = torch.randn((23, 80, 160, 160), requires_grad=False)
    # print("dim: ", fake_data.dim())
    x = test(fake_data, fake_label)
    print(type(x), x)
