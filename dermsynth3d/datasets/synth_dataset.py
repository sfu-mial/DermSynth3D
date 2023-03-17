import os
import numpy as np
import pandas as pd
from PIL import Image
import uuid

from dermsynth3d.datasets.datasets import ImageDataset

from dermsynth3d.utils.utils import (
    make_masks,
    blend_background,
    random_resize_crop_seg_lesion,
)
from dermsynth3d.utils.tensor import window_overlap_mask
from dermsynth3d.deepblend.utils import make_canvas_mask
from dermsynth3d.deepblend.blend import paste_blend


class SynthesizeDataset(ImageDataset):
    def __init__(
        self,
        dir_root,
        image_extension=".png",
        target_extension=".npz",
    ):
        self.dir_root = dir_root

        # Directories to save images and targets.
        dir_images = os.path.join(self.dir_root, "images")
        dir_targets = os.path.join(self.dir_root, "targets")
        dir_depth = os.path.join(self.dir_root, "depth")
        self.csv_filename = os.path.join(self.dir_root, "data.csv")

        # Create if not exists.
        if not os.path.isdir(self.dir_root):
            os.makedirs(self.dir_root)
        if not os.path.isdir(dir_images):
            os.makedirs(dir_images)
        if not os.path.isdir(dir_targets):
            os.makedirs(dir_targets)
        if not os.path.isdir(dir_depth):
            os.makedirs(dir_depth)

        super().__init__(
            dir_images=dir_images,
            dir_targets=dir_targets,
            dir_depth=dir_depth,
            name="Synth",
            dir_predictions=None,
            image_extension=image_extension,
            target_extension=target_extension,
            image_augment=None,
            image_preprocess=None,
            target_preprocess=None,
        )

        self.blend_df = pd.DataFrame([])
        if os.path.isfile(self.csv_filename):
            # Load data-frame if already exists.
            self.blend_df = pd.read_csv(self.csv_filename)

        self.all_params = []

    def generate_target_name(self):
        return uuid.uuid4().hex

    def target(self, image_id):
        target = np.load(self.target_filepath(image_id))
        target = target["masks"]
        return target

    def save_image(self, target_id: str, img: np.array):
        im = Image.fromarray(img)
        im.save(self.image_filepath(target_id))

    def save_depth(self, target_id: str, img: np.array):
        im = Image.fromarray(img)
        im.save(self.depth_filepath(target_id))

    def save_target(self, target_id: str, masks: np.array):
        """Saves a compressed numpy array of the target variables."""
        np.savez_compressed(self.target_filepath(target_id), masks=masks)

    def update_params(self, params):
        self.all_params.append(params)

    def save_params(self):
        new_df = pd.DataFrame.from_records(self.all_params)
        extended_df = pd.concat((self.blend_df, new_df), ignore_index=True)
        extended_df.to_csv(self.csv_filename, index=False)
        self.blend_df = extended_df
