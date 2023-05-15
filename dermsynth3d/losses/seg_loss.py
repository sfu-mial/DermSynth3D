import numpy as np
import torch


def dice_loss(target, input):
    """
    Args:

    target: [BxCxHxW]
    input: [BxCxHxW]

    Returns:

    loss: [float]
    """
    smooth = 0.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    if (tflat.sum() == 0) and (iflat.sum() < 1):
        return 0

    loss = 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

    return loss


def loss_batch_channel(segs, preds, loss_func):
    """Compute the loss for each sample and channel in the batch."""
    loss = 0
    count_batch_channels = 0
    for batch_idx in np.arange(segs.shape[0]):
        pred = preds[batch_idx]
        seg = segs[batch_idx]

        for channel_idx in np.arange(seg.shape[0]):
            loss_channel = loss_func(seg[channel_idx], pred[channel_idx])
            if loss_channel is not None:
                loss += loss_channel
                count_batch_channels += 1

    if count_batch_channels > 0:
        loss /= count_batch_channels

    return loss


def jaccard_loss(targets, preds, empty_union=None, ignore_empty_target=True):
    """
    Computes Jaccard Loss
    """
    if ignore_empty_target:
        if targets.sum() == 0:
            return None

    if (targets.sum() == 0) and (preds.sum() == 0):
        return empty_union

    intersect = torch.minimum(preds, targets)
    union = torch.maximum(preds, targets)

    return 1 - intersect.sum() / union.sum()
