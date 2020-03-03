import torch.nn as nn
import torch


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)


def compute_per_channel_dice(input_data, target, epsilon=1e-5, ignore_index=None, weight=None):
    """
    :param input_data: tensor
    :param target: tensor
    :param epsilon:
    :param ignore_index:
    :param weight:
    :return: (tensor)
    """
    # assumes that input is a normalized probability
    # input and target shapes must match
    assert input_data.size() == target.size(), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False

        input_data = input_data * mask
        target = target * mask

    input_data = flatten(input_data)
    target = flatten(target)

    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input_data * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input_data + target).sum(-1)
    return 2.0 * intersect / denominator.clamp(min=epsilon)


class DiceCoefficient1(nn.Module):
    def __init__(self):
        super(DiceCoefficient1, self).__init__()

    def forward(self, prediction, true):
        b = prediction.size(1)
        coefficient_list = {}

        prediction = prediction.view(b, -1)
        true = true.view(b, -1)

        for i in range(b):
            p = prediction[i, :].unsqueeze(0)
            t = true[i, :].unsqueeze(0)
            coefficient = 2 * (p * t).sum(1) / (p.sum(1) + t.sum(1))
            coefficient_list['{}'.format(i)]= coefficient
        return coefficient_list


class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-5, ignore_index=None, classes_mean=True, **kwargs):
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.classes_mean = classes_mean

    def __call__(self, input_data, target):
        """
        :param input_data: 5D probability maps torch tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: Soft Dice Coefficient averaged over all channels/classes
        """
        # Average across channels in order to get the final score
        classes_mean = self.classes_mean
        if classes_mean:
            return torch.mean(compute_per_channel_dice(input_data, target, epsilon=self.epsilon, ignore_index=self.ignore_index)[1:])
        else:
            return compute_per_channel_dice(input_data, target, epsilon=self.epsilon, ignore_index=self.ignore_index)[1:]


if __name__ == '__main__':
    test = DiceCoefficient(classes_mean=False)
    rest1 = DiceCoefficient1()
    fake_data = torch.randn((10, 23, 64, 128, 128), requires_grad=False)
    fake_label = torch.randn((10, 23, 64, 128, 128), requires_grad=False)

    x = test(fake_data, fake_label)
    y = rest1(fake_data, fake_label)
    print(x)
    print(y)
