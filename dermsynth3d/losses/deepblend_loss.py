import torch
from dermsynth3d.deepblend.utils import laplacian_filter_tensor, gram_matrix

"""
Some parts of the code borrowed from: https://github.com/owenzlz/DeepImageBlending
"""


def style_gram_loss(target_features_style, blend_features_style, loss_func):
    """
    Computes Gram loss for style
    """
    target_gram_style = [gram_matrix(y) for y in target_features_style]
    blend_gram_style = [gram_matrix(y) for y in blend_features_style]

    style_loss = 0
    for layer in range(len(blend_gram_style)):
        style_loss += loss_func(blend_gram_style[layer], target_gram_style[layer])

    style_loss /= len(blend_gram_style)
    return style_loss


def total_variation_loss(blend_img_tensor):
    """
    Compute TV Reg Loss
    """
    tv_loss = torch.sum(
        torch.abs(blend_img_tensor[:, :, :, :-1] - blend_img_tensor[:, :, :, 1:])
    ) + torch.sum(
        torch.abs(blend_img_tensor[:, :, :-1, :] - blend_img_tensor[:, :, 1:, :])
    )
    return tv_loss


def gradient_loss(blend_img_tensor, gt_gradient, loss_func, gpu_id):
    """
    Compute Laplacian Gradient of Blended Image
    """
    pred_gradient = laplacian_filter_tensor(blend_img_tensor, gpu_id)

    # Compute Gradient Loss
    grad_loss = 0
    for c in range(len(pred_gradient)):
        grad_loss += loss_func(pred_gradient[c], gt_gradient[c])

    grad_loss /= len(pred_gradient)
    return grad_loss
