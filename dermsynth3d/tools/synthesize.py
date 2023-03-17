import os
import uuid
import numpy as np
import pandas as pd

from PIL import Image

from dermsynth3d.datasets.datasets import ImageDataset

from dermsynth3d.utils.utils import (
    make_masks,
    blend_background,
    random_resize_crop_seg_lesion,
)
from dermsynth3d.utils.tensor import window_overlap_mask
from dermsynth3d.deepblend.utils import make_canvas_mask
from dermsynth3d.deepblend.blend import paste_blend


class Synthesize2D:
    """
    Base class to generate lesion-blended renderings with random backgrounds.
    """

    def __init__(
        self,
        view2d,
        body_mask,
        skin_mask,
        lesion_mask,
        background_img,
        paste_lesion_img,
        paste_lesion_mask,
        soft_blend=False,
    ):
        self.view2d = view2d
        self.body_mask = body_mask
        self.skin_mask = skin_mask
        self.lesion_mask = lesion_mask
        self.skin_or_lesion_mask = skin_mask | lesion_mask
        self.background_img = background_img
        self.paste_lesion_img = paste_lesion_img
        self.paste_lesion_mask = paste_lesion_mask

        self.pasted_back_image = blend_background(
            self.view2d,
            self.background_img,
            self.body_mask,
            soft_blend=soft_blend,  # If True, soft blends the foreground with background.
        )

    def min_max_scale_foreground(self):
        img_shape = self.paste_lesion_img.shape
        count_fit = window_overlap_mask(
            self.skin_or_lesion_mask,
            window_size=(img_shape[0] + 2, img_shape[1] + 2),
            pad_value=0,
            output_type="count",
        )

        n_pixels = img_shape[0] * img_shape[1]

        min_scale = min_scale_size(img_shape[:2])
        if count_fit.max() == 0:
            max_scale = 1
        else:
            max_scale = np.sqrt(n_pixels / count_fit.max())

        return min_scale, max_scale

    def random_resize_foreground(self, min_scale, max_scale):
        lesion_crop, mask_crop = random_resize_crop_seg_lesion(
            self.paste_lesion_mask,
            self.paste_lesion_img,
            min_scale=min_scale,
            max_scale=max_scale,
            maintain_aspect=True,
        )

        return lesion_crop, mask_crop

    def paste_locations(self, window_size):
        overlap_with_skin = window_overlap_mask(
            self.skin_or_lesion_mask,
            window_size=(window_size[0] + 2, window_size[1] + 2),
            pad_value=0,
            output_type="all_ones",
        )
        paste_locs = np.where(overlap_with_skin)
        return paste_locs

    def paste_foreground(self, x, y, foreground_img, foreground_mask):
        pasted_foreground_img = paste_blend(
            x,
            y,
            foreground_img,
            foreground_mask[:, :, np.newaxis],
            self.pasted_back_image,
        )
        lesion_mask_paste = make_canvas_mask(
            x, y, pasted_foreground_img.shape[:2], foreground_mask
        )
        lesion_mask_paste = np.asarray(lesion_mask_paste > 0)
        lesion_mask_with_paste = lesion_mask_paste | self.lesion_mask
        masks = make_masks(lesion_mask_with_paste, self.skin_mask)
        return pasted_foreground_img, masks

    def random_paste_foreground(self, min_scale, max_scale):
        pasted_img = None
        masks = None
        foreground_resized_img, foreground_resized_mask = self.random_resize_foreground(
            min_scale, max_scale
        )

        if foreground_resized_img is not None:
            paste_locs = self.paste_locations(foreground_resized_img.shape)
            if len(paste_locs[0]) > 0:
                # Randomly choose a place to paste.
                loc_idx = np.random.randint(0, len(paste_locs[0]))
                x = paste_locs[0][loc_idx]
                y = paste_locs[1][loc_idx]
                pasted_img, masks = self.paste_foreground(
                    x, y, foreground_resized_img, foreground_resized_mask
                )

        return pasted_img, masks

    def random_paste_foreground_with_retry(
        self, min_scale, max_scale, print_debug=True
    ):
        pasted_img, masks = self.random_paste_foreground(min_scale, max_scale)

        if pasted_img is None:
            # The min max scale created an image less
            # than the min accepted image size.
            # Or there are no valid regions to paste.
            # Retry the resize, so try again with scaling
            # the max by a factor of 2.
            if print_debug:
                print(
                    "Unable to find location to paste lesion. "
                    "Trying again with a different scale"
                )
            pasted_img, masks = self.random_paste_foreground(min_scale, max_scale * 2)

            if print_debug:
                if pasted_img is None:
                    print("Failed: Could not paste. Skip this image.")
                else:
                    print("Success: Pasted with double max scale.")

        return pasted_img, masks


def min_scale_size(img_size):
    "Hardcoded min sizes based on the image size."
    if min(img_size[:2]) < 30:
        min_scale = 2
    elif min(img_size[:2]) < 50:
        min_scale = 5
    elif min(img_size[:2]) < 100:
        min_scale = 10
    elif min(img_size[:2]) < 150:
        min_scale = 15
    else:
        min_scale = 20

    return min_scale
