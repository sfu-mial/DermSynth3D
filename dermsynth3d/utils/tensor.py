from functools import reduce
from operator import __add__

import PIL

import torch
import numpy as np
import torch.nn.functional as F


def same_pad_tensor(x: torch.Tensor, kernel_size: np.array, value: int):
    conv_padding = reduce(
        __add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]]
    )

    x_padd = F.pad(x, conv_padding, value=value)
    return x_padd


def window_overlap_mask(
    mask,
    window_size,
    pad_value: int = 0,
    output_type: str = "all_ones",
):
    """
    Pad the mask at the borders to maintain original size.
    Where the `window_size` has all 1's in the `mask`.
    """
    pad_mask = same_pad_tensor(torch.Tensor(mask), window_size, value=pad_value)

    avg_pool = torch.nn.AvgPool2d(window_size, stride=(1, 1))
    overlap = avg_pool(pad_mask[np.newaxis, :, :])
    overlap = overlap.squeeze().numpy()
    if output_type == "all_ones":
        overlap = overlap >= 1
    elif output_type == "count":
        overlap = overlap * window_size[0] * window_size[1]
    else:
        raise NotImplementedError(
            "Error: `output_type={}` is not supported".format(output_type)
        )

    return overlap


def max_value_in_window(img, window_size, pad_value=-1):
    pad_img = same_pad_tensor(torch.Tensor(img), window_size, value=pad_value)

    max_pool = torch.nn.MaxPool2d(window_size, stride=(1, 1))
    max_window_img = max_pool(pad_img[np.newaxis, :, :])
    return max_window_img


def pil_to_tensor(img: PIL.Image.Image):
    img_np = (np.asarray(img) / 255).astype(np.float32)
    img_tensor = torch.tensor(img_np, dtype=torch.float32)#.cuda()
    return img_tensor


def diff_max_min(x, kernel_size):
    maxpool = torch.nn.MaxPool2d(tuple(kernel_size), stride=(1, 1))
    maxVals = maxpool((x))
    minVals = maxpool((x * -1)) * -1

    diff_patch = torch.abs(maxVals - minVals)  # .numpy()
    diff_img = diff_patch.mean(axis=0).squeeze()
    return diff_img


def depth_differences(
    x, patch_kernel_size, local_kernel_size: tuple = (3, 3), value: int = -1
):
    x_depth_local = same_pad_tensor(x, local_kernel_size, value=value)
    local_depth_max_diff = diff_max_min(x_depth_local, local_kernel_size)

    x_depth_patch = same_pad_tensor(
        local_depth_max_diff.unsqueeze(0), patch_kernel_size, value=value
    )
    patch_depth_max_diff = diff_max_min(x_depth_patch, patch_kernel_size)
    return patch_depth_max_diff


def augmented_predictions(
    num_augmentations: int,
    test_img,
    seg_model,
    img_augment_func,
    img_transform_func,
    device,
):
    softmax2d = torch.nn.Softmax2d()
    test_img = np.asarray(test_img)
    preds_out = []
    for _ in np.arange(num_augmentations):
        # aug = spatial_augment(image=test_img)
        aug = img_augment_func(image=test_img)
        pred_out = seg_model(img_transform_func(aug["image"])[None, :].to(device))[
            "out"
        ]
        pred_out = softmax2d(pred_out)
        preds_out.append(pred_out.cpu().detach().numpy())

    preds_out = np.asarray(preds_out)
    preds_out = preds_out.squeeze()
    roll_out = np.rollaxis(preds_out, 1, 4)
    return roll_out


def average_augmented_predictions(aug_preds, img_size: tuple):
    channels = 3
    preds_avg = np.zeros(shape=(img_size[0], img_size[1], channels))
    preds_avg[:, :, 0] = aug_preds.mean(axis=0)[:, :, 0]
    preds_avg[:, :, 1] = aug_preds.mean(axis=0)[:, :, 1]
    preds_avg[:, :, 2] = aug_preds.mean(axis=0)[:, :, 2]
    return preds_avg
