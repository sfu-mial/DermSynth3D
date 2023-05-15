from PIL import Image
import numpy as np


def load_image(img_path, img_size=None, mode="RGB", resample=Image.NEAREST):
    """Common function to load, convert to a mode, and resize an image.

    Args:
        img_path (string): Name and path to the image.
        img_size (tuple, optional): W x H of the image.
            If `img_size=None` then use original size.
            Else, resize to the target size.
            Defaults to None.
        mode (string, optional): Conversion mode of the image.
            See:
            https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert


    Returns:
        [PIL image]: The image from `img_path`.
    """
    img = Image.open(img_path).convert(mode)
    if img_size is not None:
        img = img.resize(img_size, resample)

    return img


def crop_amount(img_size, min_pad=8, increment=128):
    largest_img_size = np.max(img_size)
    crop_amount = np.maximum(min_pad, min_pad / increment * largest_img_size)
    crop_amount = np.ceil(crop_amount).astype(int)
    return crop_amount


def simple_augment(lesion_img, lesion_seg):
    if np.random.random_sample() > 0.5:
        lesion_img = np.fliplr(lesion_img)
        lesion_seg = np.fliplr(lesion_seg)

    if np.random.random_sample() > 0.5:
        lesion_img = np.flipud(lesion_img)
        lesion_seg = np.flipud(lesion_seg)

    if np.random.random_sample() > 0.5:
        lesion_img = np.rot90(lesion_img)
        lesion_seg = np.rot90(lesion_seg)

    return lesion_img, lesion_seg


def uint8_to_float32(x):
    if x.dtype != np.uint8:
        raise ValueError("Expects dtype=uint8")

    if x.max() <= 1:
        raise ValueError("Error: x.max() <= 1, are you sure you want to convert?")

    return (x / 255).astype(np.float32)


def float_img_to_uint8(img):
    if (img.dtype != np.float32) and (img.dtype != np.float64):
        raise ValueError("Expects dtype=float32")

    if img.max() > 1:
        raise ValueError("Expects img.max() <= 1")

    if img.min() < 0:
        raise ValueError("Expects img.min() >= 0")

    return (img * 255).astype(np.uint8)
