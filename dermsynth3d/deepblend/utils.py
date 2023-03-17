import numpy as np
import torch

"""
Some parts of the code borrowed from: https://github.com/owenzlz/DeepImageBlending
"""


def single_channel_to_rgb_tensor(x, gpu_id=0):
    """
    Convert an MxN array into 1 x 3 x M x N tensor.
    """
    x_tensor = numpy2tensor(x, gpu_id=gpu_id)
    x_rgb = (
        x_tensor.squeeze(0).repeat(3, 1).view(3, x.shape[0], x.shape[1]).unsqueeze(0)
    )

    return x_rgb


def make_canvas_mask(x_start, y_start, target_size, mask):
    """
    Args:
        x_start ([type]): [description]
        y_start ([type]): [description]
        target_size ([type]): [description]
        mask ([type]): [description]

    Returns:
        [type]: [description]
    """
    canvas_mask = np.zeros(target_size)
    canvas_mask[
        int(x_start - mask.shape[0] * 0.5) : int(x_start + mask.shape[0] * 0.5),
        int(y_start - mask.shape[1] * 0.5) : int(y_start + mask.shape[1] * 0.5),
    ] = mask
    return canvas_mask


def numpy2tensor(np_array, gpu_id=0, is_contiguous=False):
    """
    Converts numpy array to torch tensor
    """
    if len(np_array.shape) == 2:
        if is_contiguous:
            raise NotImplementedError("Not tested for 2D inputs.")
        else:
            tensor = torch.from_numpy(np_array).unsqueeze(0).float().to(gpu_id)
    else:
        if is_contiguous:
            tensor = (
                torch.from_numpy(np_array)
                .unsqueeze(0)
                .transpose(1, 3)
                .transpose(2, 3)
                .float()
                .contiguous()
                .to(gpu_id)
            )
        else:
            tensor = (
                torch.from_numpy(np_array)
                .unsqueeze(0)
                .transpose(1, 3)
                .transpose(2, 3)
                .float()
                .to(gpu_id)
            )
    return tensor


def laplacian_filter_tensor(img_tensor, gpu_id):
    """
    Applies Laplalican Operator per image channel
    """
    laplacian_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian_conv = torch.nn.Conv2d(
        1, 1, kernel_size=3, stride=1, padding=1, bias=False
    )
    laplacian_conv.weight = torch.nn.Parameter(
        torch.from_numpy(laplacian_filter).float().unsqueeze(0).unsqueeze(0).to(gpu_id)
    )

    for param in laplacian_conv.parameters():
        param.requires_grad = False

    red_img_tensor = img_tensor[:, 0, :, :].unsqueeze(1)
    green_img_tensor = img_tensor[:, 1, :, :].unsqueeze(1)
    blue_img_tensor = img_tensor[:, 2, :, :].unsqueeze(1)

    red_gradient_tensor = laplacian_conv(red_img_tensor).squeeze(1)
    green_gradient_tensor = laplacian_conv(green_img_tensor).squeeze(1)
    blue_gradient_tensor = laplacian_conv(blue_img_tensor).squeeze(1)
    return red_gradient_tensor, green_gradient_tensor, blue_gradient_tensor


def compute_gt_gradient(x_start, y_start, source_img, target_img, mask, gpu_id):
    # compute source image gradient
    source_img_tensor = (
        torch.from_numpy(source_img)
        .unsqueeze(0)
        .transpose(1, 3)
        .transpose(2, 3)
        .float()
        .to(gpu_id)
    )
    (
        red_source_gradient_tensor,
        green_source_gradient_tensor,
        blue_source_gradient_tenosr,
    ) = laplacian_filter_tensor(source_img_tensor, gpu_id)
    red_source_gradient = red_source_gradient_tensor.cpu().data.numpy()[0]
    green_source_gradient = green_source_gradient_tensor.cpu().data.numpy()[0]
    blue_source_gradient = blue_source_gradient_tenosr.cpu().data.numpy()[0]

    # compute target image gradient
    target_img_tensor = (
        torch.from_numpy(target_img)
        .unsqueeze(0)
        .transpose(1, 3)
        .transpose(2, 3)
        .float()
        .to(gpu_id)
    )
    (
        red_target_gradient_tensor,
        green_target_gradient_tensor,
        blue_target_gradient_tenosr,
    ) = laplacian_filter_tensor(target_img_tensor, gpu_id)
    red_target_gradient = red_target_gradient_tensor.cpu().data.numpy()[0]
    green_target_gradient = green_target_gradient_tensor.cpu().data.numpy()[0]
    blue_target_gradient = blue_target_gradient_tenosr.cpu().data.numpy()[0]

    # mask and canvas mask
    canvas_mask = np.zeros((target_img.shape[0], target_img.shape[1]))
    canvas_mask[
        int(x_start - source_img.shape[0] * 0.5) : int(
            x_start + source_img.shape[0] * 0.5
        ),
        int(y_start - source_img.shape[1] * 0.5) : int(
            y_start + source_img.shape[1] * 0.5
        ),
    ] = mask

    # foreground gradient
    red_source_gradient = red_source_gradient * mask
    green_source_gradient = green_source_gradient * mask
    blue_source_gradient = blue_source_gradient * mask
    red_foreground_gradient = np.zeros((canvas_mask.shape))
    red_foreground_gradient[
        int(x_start - source_img.shape[0] * 0.5) : int(
            x_start + source_img.shape[0] * 0.5
        ),
        int(y_start - source_img.shape[1] * 0.5) : int(
            y_start + source_img.shape[1] * 0.5
        ),
    ] = red_source_gradient
    green_foreground_gradient = np.zeros((canvas_mask.shape))
    green_foreground_gradient[
        int(x_start - source_img.shape[0] * 0.5) : int(
            x_start + source_img.shape[0] * 0.5
        ),
        int(y_start - source_img.shape[1] * 0.5) : int(
            y_start + source_img.shape[1] * 0.5
        ),
    ] = green_source_gradient
    blue_foreground_gradient = np.zeros((canvas_mask.shape))
    blue_foreground_gradient[
        int(x_start - source_img.shape[0] * 0.5) : int(
            x_start + source_img.shape[0] * 0.5
        ),
        int(y_start - source_img.shape[1] * 0.5) : int(
            y_start + source_img.shape[1] * 0.5
        ),
    ] = blue_source_gradient

    # background gradient
    red_background_gradient = red_target_gradient * (canvas_mask - 1) * (-1)
    green_background_gradient = green_target_gradient * (canvas_mask - 1) * (-1)
    blue_background_gradient = blue_target_gradient * (canvas_mask - 1) * (-1)

    # add up foreground and background gradient
    gt_red_gradient = red_foreground_gradient + red_background_gradient
    gt_green_gradient = green_foreground_gradient + green_background_gradient
    gt_blue_gradient = blue_foreground_gradient + blue_background_gradient

    gt_red_gradient = numpy2tensor(gt_red_gradient, gpu_id)
    gt_green_gradient = numpy2tensor(gt_green_gradient, gpu_id)
    gt_blue_gradient = numpy2tensor(gt_blue_gradient, gpu_id)

    gt_gradient = [gt_red_gradient, gt_green_gradient, gt_blue_gradient]
    return gt_gradient


def gram_matrix(y):
    # Compute Gram matrix
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def get_matched_features_pytorch(blended_features, target_features):
    matched_features = blended_features.new_full(
        size=blended_features.size(), fill_value=0, requires_grad=False
    ).to(blended_features.device)
    for filter in range(0, blended_features.size(1)):
        matched_filter = hist_match_pytorch(
            blended_features[0, filter, :, :], target_features[0, filter, :, :]
        )
        matched_features[0, filter, :, :] = matched_filter
    return matched_features


def hist_match_pytorch(source, template):
    oldshape = source.size()
    source = source.view(-1)
    template = template.view(-1)

    max_val = max(source.max().item(), template.max().item())
    min_val = min(source.min().item(), template.min().item())

    num_bins = 400
    hist_step = (max_val - min_val) / num_bins

    if hist_step == 0:
        return source.reshape(oldshape)

    hist_bin_centers = torch.arange(start=min_val, end=max_val, step=hist_step).to(
        source.device
    )
    hist_bin_centers = hist_bin_centers + hist_step / 2.0

    source_hist = torch.histc(input=source, min=min_val, max=max_val, bins=num_bins)
    template_hist = torch.histc(input=template, min=min_val, max=max_val, bins=num_bins)

    source_quantiles = torch.cumsum(input=source_hist, dim=0)
    source_quantiles = source_quantiles / source_quantiles[-1]

    template_quantiles = torch.cumsum(input=template_hist, dim=0)
    template_quantiles = template_quantiles / template_quantiles[-1]

    nearest_indices = torch.argmin(
        torch.abs(
            template_quantiles.repeat(len(source_quantiles), 1)
            - source_quantiles.view(-1, 1).repeat(1, len(template_quantiles))
        ),
        dim=1,
    )

    source_bin_index = torch.clamp(
        input=torch.round(source / hist_step), min=0, max=num_bins - 1
    ).long()

    mapped_indices = torch.gather(input=nearest_indices, dim=0, index=source_bin_index)
    matched_source = torch.gather(input=hist_bin_centers, dim=0, index=mapped_indices)

    return matched_source.reshape(oldshape)
