import cv2
import numpy as np


def shade_of_gray_cc(img, power=6, gamma=None):
    """
    Code from:
    https://www.kaggle.com/code/apacheco/shades-of-gray-color-constancy/notebook

    img (numpy array): the original image with format of (h, w, c)
    power (int): the degree of norm, 6 is used in reference paper
    gamma (float): the value of gamma correction, 2.2 is used in reference paper
    """

    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype("uint8")
        look_up_table = np.ones((256, 1), dtype="uint8") * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i / 255, 1 / gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype("float32")
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    img = np.clip(img, a_min=0, a_max=255)

    return img.astype(img_dtype)
