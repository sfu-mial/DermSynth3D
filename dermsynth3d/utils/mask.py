import numpy as np
from dermsynth3d.utils.image import (
    crop_amount,
    uint8_to_float32,
)


def can_blend_mask(mask, min_pad=8):
    """Return True if the masked lesion meets blending requirements.

    Args:
        mask (_type_): _description_
        min_pad (int, optional): _description_. Defaults to 8.

    Returns:
        _type_: _description_
    """
    # Don't blend if all zeros.
    # ie., there is no lesion.
    if np.max(mask) == 0:
        return False

    x, y = np.where(mask)
    crop_mask = mask[x.min() : x.max(), y.min() : y.max()]
    pad = crop_amount(crop_mask.shape, min_pad=min_pad)
    start_x = x.min() - pad
    end_x = x.max() + pad
    start_y = y.min() - pad
    end_y = y.max() + pad

    if start_x < 0:
        return False

    if end_x > mask.shape[0]:
        return False

    if start_y < 0:
        return False

    if end_y > mask.shape[1]:
        return False

    return True


def box_crop_lesion(
    img,
    mask,
    force_even_dims: bool = True,
    asfloat: bool = True,
):
    if mask.sum() == 0:
        # No lesion in mask.
        raise ValueError("Selected lesion mask is empty")

    x, y = np.where(mask)

    crop_mask = mask[x.min() : x.max(), y.min() : y.max()]

    # A certain amount of padding is needed to compute the gradients
    # around the border. This will pad at least 8 pixels, but
    # can be more if the cropped lesion is large.
    pad = crop_amount(crop_mask.shape)
    # Doesn't check for out of bounds with padding.
    start_x = x.min() - pad
    end_x = x.max() + pad
    start_y = y.min() - pad
    end_y = y.max() + pad

    if force_even_dims:
        if (end_x - start_x) % 2:
            start_x = start_x - 1
        if (end_y - start_y) % 2:
            start_y = start_y - 1

    img_crop = img[start_x:end_x, start_y:end_y, :]
    mask_crop = mask[start_x:end_x, start_y:end_y]

    if asfloat:
        img_crop = uint8_to_float32(img_crop)

    return img_crop, mask_crop
