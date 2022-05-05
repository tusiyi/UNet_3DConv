import torch
from torch import Tensor


def dice_coeff(input_: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input_.size() == target.size()
    if input_.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input_.shape})')

    if input_.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input_.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input_) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input_.shape[0]):
            dice += dice_coeff(input_[i, ...], target[i, ...])
        return dice / input_.shape[0]


def multiclass_dice_coeff(input_: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input_.size() == target.size()
    dice = 0
    for channel in range(input_.shape[1]):
        dice += dice_coeff(input_[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input_.shape[1]


def dice_loss(input_: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input_.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input_, target, reduce_batch_first=True)
