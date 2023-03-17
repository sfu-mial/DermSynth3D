import os
import json
import cv2
import random

import numpy as np
import pandas as pd

from PIL import Image, ImageDraw, ImageFilter
from typing import Optional, List
from collections.abc import Callable

from skimage.measure import label
import torch
from torchvision.ops import masks_to_boxes

from dermsynth3d.utils.image import (
    load_image,
    uint8_to_float32,
    crop_amount,
    # float_img_to_uint8,
)
from dermsynth3d.utils.annotate import (
    poly_from_xy,
)
from dermsynth3d.utils.channels import Target
from dermsynth3d.utils.mask import (
    can_blend_mask,
    box_crop_lesion,
)


class ImageDataset:
    def __init__(
        self,
        dir_images: str,
        dir_targets: str,
        dir_depth: Optional[str] = None,
        name: str = "dataset",
        dir_predictions: Optional[str] = None,
        image_extension: str = ".png",
        target_extension: Optional[str] = ".png",
        spatial_transform: Optional[Callable] = None,
        image_augment=None,
        image_preprocess=None,
        target_preprocess=None,
        totensor=None,
        color_constancy=None,
    ):
        """
        Base class to load 2D image datasets.
        """

        self.dir_images = dir_images
        self.dir_targets = dir_targets
        self.dir_depth = dir_depth
        self.name = name
        self.dir_predictions = dir_predictions

        self.image_extension = image_extension
        self.target_extension = target_extension

        self.spatial_transform = spatial_transform
        self.image_augment = image_augment
        self.image_preprocess = image_preprocess
        self.target_preprocess = target_preprocess
        self.totensor = totensor
        self.color_constancy = color_constancy

        self.file_ids = self.file_ids()

    def __len__(self):
        return len(self.file_ids)

    def file_ids(self) -> List[str]:
        """Return a list of the unique file identifiers."""
        return [fname.split(".")[0] for fname in os.listdir(self.dir_images)]

    def image_filepath(self, image_id):
        return os.path.join(self.dir_images, image_id + self.image_extension)

    def depth_filepath(self, image_id):
        return os.path.join(self.dir_depth, image_id + self.image_extension)

    def image(
        self, image_id: str, img_size: Optional[tuple] = None, resample=Image.BILINEAR
    ):
        """Returns a PIL image for the given `image_id`.

        Args:
            image_id (string): _description_
            img_size (Optional[tuple], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        img = load_image(
            self.image_filepath(image_id), img_size=img_size, resample=resample
        )
        if self.color_constancy:
            img = self.color_constancy(np.asarray(img))
            img = Image.fromarray(img)

        return img

    def target_filepath(self, image_id):
        return os.path.join(self.dir_targets, image_id + self.target_extension)

    def target(self, image_id, img_size: Optional[tuple] = None):
        return load_image(self.target_filepath(image_id), img_size)

    def prediction_filepath(self, image_id):
        return os.path.join(self.dir_predictions, image_id + self.target_extension)

    def prediction(self, image_id):
        return load_image(self.prediction_filepath(image_id))

    def bbox_from_target(self, mask):
        labels = label(mask, background=0, connectivity=2)
        mask = torch.tensor(labels).unsqueeze(0)

        obj_ids = torch.unique(mask)

        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        boxes = masks_to_boxes(masks)

        boxes = boxes[boxes[:, 0] < boxes[:, 2]]
        boxes = boxes[boxes[:, 1] < boxes[:, 3]]

        return boxes

    def get_target_from_mask(self, mask):
        mask_labels = label(mask, background=0, connectivity=2)
        mask = torch.tensor(mask_labels).unsqueeze(0)

        obj_ids = torch.unique(mask)

        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        boxes = masks_to_boxes(masks)

        boxes = boxes[boxes[:, 0] < boxes[:, 2]]
        boxes = boxes[boxes[:, 1] < boxes[:, 3]]

        # there is only one class
        labels = torch.ones((masks.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return target

    def __getitem__(self, idx: int):
        file_id = self.file_ids[idx]
        img = np.asarray(self.image(file_id))
        target = np.asarray(self.target(file_id))

        if self.spatial_transform is not None:
            spatial_transformed = self.spatial_transform(
                image=img,
                mask=target,
            )

            img = spatial_transformed["image"]
            target = spatial_transformed["mask"]

        if self.image_augment is not None:
            img_transformed = self.image_augment(image=img)
            img = img_transformed["image"]

        if self.image_preprocess:
            preprocess_image = self.image_preprocess(image=img)
            img = preprocess_image["image"]

        if self.totensor:
            tensors = self.totensor(image=img, mask=target)
            img = tensors["image"]
            target = tensors["mask"]

        return self.name, file_id, img, target


class RealDataset_Detection(ImageDataset):
    def bbox(self, image_id):
        mask = np.asarray(self.target(image_id))[:, :, 0]
        return self.bbox_from_target(mask)

    def __getitem__(self, idx: int):
        file_id = self.file_ids[idx]
        img = np.asarray(self.image(file_id))
        mask = np.asarray(self.target(file_id))

        if self.spatial_transform is not None:
            spatial_transformed = self.spatial_transform(
                image=img,
                mask=mask,
            )

            img = spatial_transformed["image"]
            mask = spatial_transformed["mask"]

        if self.image_augment is not None:
            img_transformed = self.image_augment(image=img)
            img = img_transformed["image"]

        if self.image_preprocess:
            preprocess_image = self.image_preprocess(image=img)
            img = preprocess_image["image"]

        target = self.get_target_from_mask(mask[:, :, 0])
        if self.totensor:
            tensors = self.totensor(image=img, mask=mask)
            img = tensors["image"]
            mask = tensors["mask"]

        return self.name, file_id, img, mask, target


class SynthDataset_Detection(ImageDataset):
    def target(self, image_id):
        target = np.load(self.target_filepath(image_id))
        target = target["masks"]
        return target

    def save_image(self, target_id: str, img: np.array):
        im = Image.fromarray(img)
        im.save(self.image_filepath(target_id))

    def save_target(self, target_id: str, masks: np.array):
        np.savez_compressed(self.target_filepath(target_id), masks=masks)

    def bbox(self, image_id):
        mask = self.target(image_id)[:, :, 0]
        return self.bbox_from_target(mask)

    def __getitem__(self, idx: int):
        file_id = self.file_ids[idx]
        img = np.asarray(self.image(file_id))
        mask = np.asarray(self.target(file_id))

        if self.spatial_transform is not None:
            spatial_transformed = self.spatial_transform(
                image=img,
                mask=mask,
            )

            img = spatial_transformed["image"]
            mask = spatial_transformed["mask"]

        if self.image_augment is not None:
            img_transformed = self.image_augment(image=img)
            img = img_transformed["image"]

        if self.image_preprocess:
            preprocess_image = self.image_preprocess(image=img)
            img = preprocess_image["image"]

        target = self.get_target_from_mask(mask[:, :, 0])
        if self.totensor:
            tensors = self.totensor(image=img, mask=mask)
            img = tensors["image"]
            mask = tensors["mask"]

        return self.name, file_id, img, mask, target


class PratheepanSkinDataset(ImageDataset):
    """
    GT loader for Pratheepan Skin Dataset
    """

    def target(self, image_id, img_size: Optional[tuple] = None):
        target = load_image(self.target_filepath(image_id), img_size)
        target = np.asarray(target) / 255
        return target[:, :, 0]


class BinarySegementationDataset(ImageDataset):
    """
    GT loader for Binary Segmentation
    """

    def target(self, image_id, img_size: Optional[tuple] = None):
        target = load_image(self.target_filepath(image_id), img_size)
        target = np.asarray(target) / 255

        return target


class HGRDataset(ImageDataset):
    """
    GT loader for HGR Dataset
    """

    def target(self, image_id, img_size: Optional[tuple] = None):
        target = load_image(self.target_filepath(image_id), img_size)
        target = np.asarray(target)
        target = ~target
        target = target / 255

        return target


class SynthDataset(ImageDataset):
    """
    Helper function to load and savee synthesized GTs.
    """

    def target(self, image_id):
        target = np.load(self.target_filepath(image_id))
        target = target["masks"]

        return target

    def save_image(self, target_id: str, img: np.array):
        im = Image.fromarray(img)
        im.save(self.image_filepath(target_id))

    def save_target(self, target_id: str, masks: np.array):
        np.savez_compressed(self.target_filepath(target_id), masks=masks)


class FitzDataset(ImageDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        dir_images: str,
        dir_annotations: Optional[str] = None,
        image_extension: Optional[str] = ".jpg",
        spatial_transform=None,
        image_augment=None,
        image_preprocess=None,
        totensor=None,
        annotation_type="csv",
        color_constancy=None,
    ):
        self.df = df
        self.dir_annotations = dir_annotations
        self.annotation_type = annotation_type
        # All annotation IDs.
        self.all_annotation_ids = self.all_annotation_ids()
        # Only the annotations IDs that have non-empty selected lesions.
        self.annotation_ids = self.select_lesion_ids()

        super().__init__(
            dir_images=dir_images,
            dir_targets=None,
            dir_depth=None,
            name="Fitz17k",
            dir_predictions=None,
            image_extension=image_extension,
            target_extension=None,
            spatial_transform=spatial_transform,
            image_augment=image_augment,
            image_preprocess=image_preprocess,
            target_preprocess=None,
            totensor=totensor,
            color_constancy=color_constancy,
        )

    def file_ids(self) -> List[str]:
        """Return a list of the unique file identifiers."""
        if self.df is None:
            return list()

        return list(self.df.md5hash.values)

    def image_transformed(self, image_id):
        img = np.asarray(self.image(image_id))
        if self.spatial_transform:
            img_transformed = self.spatial_transform(image=img)
            img = img_transformed["image"]

        if self.image_augment:
            img_transformed = self.image_augment(image=img)
            img = img_transformed["image"]

        if self.image_preprocess:
            img_transformed = self.image_preprocess(image=img)
            img = img_transformed["image"]

        if self.totensor:
            img_transformed = self.totensor(image=img)
            img = img_transformed["image"]

        return img

    def __getitem__(self, idx: int):
        record = self.df.iloc[idx]
        img = self.image_transformed(record.md5hash)
        return (
            record.md5hash,
            record.fitzpatrick,
            img,
        )

    def all_annotation_ids(self):
        filenames = sorted(os.listdir(self.dir_annotations))
        if self.annotation_type == "csv":
            return [fname.split(".")[0] for fname in filenames]

        return [fname for fname in filenames if fname[0] != "."]

    def annotation_filepath(self, image_id):
        return os.path.join(self.dir_annotations, image_id + ".csv")

    def selected_lesion_filepath(self, image_id):
        return os.path.join(self.dir_annotations, image_id, "selected_lesion.png")

    def lesions_filepath(self, image_id):
        return os.path.join(self.dir_annotations, image_id, "lesions.png")

    def nonskin_filepath(self, image_id):
        return os.path.join(self.dir_annotations, image_id, "nonskin.png")

    def mask(self, filepath, img_size=None):
        mask = load_image(filepath, img_size, mode="L")
        mask = np.asarray(mask)
        mask = uint8_to_float32(mask)
        return mask

    def mask_nonskin(self, image_id, img_size=None):
        return self.mask(self.nonskin_filepath(image_id), img_size)

    def mask_skin(self, image_id, exclude_lesion=True, img_size=None):
        nonskin = self.mask_nonskin(image_id, img_size=img_size)
        lesions = self.mask_lesions(image_id, img_size=img_size)
        if exclude_lesion:
            skin = (1 - nonskin) * (1 - lesions)
        else:
            skin = 1 - nonskin

        return skin

    def mask_lesions(self, image_id, img_size=None):
        return self.mask(self.lesions_filepath(image_id), img_size)

    def mask_selected_lesion(self, image_id, img_size=None):
        return self.mask(self.selected_lesion_filepath(image_id), img_size)

    def annotation_df(self, image_id):
        ann_df = pd.read_csv(self.annotation_filepath(image_id))
        for f in ann_df.filename:
            assert f == image_id + ".jpg", "Error: Wrong filename"

        return ann_df

    def poly_shapes(self, image_id: str):
        shape_dict = self.poly_dict(image_id)

        poly = poly_from_xy(shape_dict["all_points_x"], shape_dict["all_points_y"])

        return poly

    def poly_dict(self, image_id: str):
        ann_df = self.annotation_df(image_id)
        assert len(ann_df) == 1, "Error: only supports 1 shape per image."

        shape_str = ann_df.region_shape_attributes.iloc[0]
        shape_dict = json.loads(shape_str)
        return shape_dict

    def poly_xy(self, image_id: str):
        shape_dict = self.poly_dict(image_id)
        x = np.asarray(shape_dict["all_points_x"])
        y = np.asarray(shape_dict["all_points_y"])
        return x, y

    def contour_image(self, image_id: str):
        img = self.image(image_id)
        poly = self.poly_shapes(image_id)
        out = cv2.drawContours(np.asarray(img), [np.asarray(poly)], -1, (0, 255, 0), 3)
        return out

    def shape_mask(self, image_id: str):
        img = self.image(image_id)
        poly = self.poly_shapes(image_id)
        poly_img = Image.new("L", img.size, 0)
        ImageDraw.Draw(poly_img).polygon(poly, outline=1, fill=255)

        return poly_img

    def box_crop_lesion(
        self,
        image_id: str,
        pad: int = 8,
        force_even_dims=False,
        asfloat=False,
    ):
        """Returns the image and mask cropped around the lesion.

        Args:
            image_id (str): _description_
            pad (int, optional): Amount to pad around the mask. Defaults to 8.
                A certain amount of padding is needed to compute the gradients
                around the border. If we go less than 8, have to check other code.
            force_even_dims (bool, optional): _description_. Defaults to False.
            asfloat (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        img = self.image(image_id)
        img = np.asarray(img)

        if self.annotation_type == "csv":
            mask = self.shape_mask(image_id)
            mask = np.asarray(mask)
            poly_x, poly_y = self.poly_xy(image_id)
        else:
            mask = self.mask_selected_lesion(image_id)
            # Flip x,y for np.where().
            poly_y, poly_x = np.where(mask)

        # Doesn't check for out of bounds with padding.
        start_x = poly_x.min() - pad
        end_x = poly_x.max() + pad
        start_y = poly_y.min() - pad
        end_y = poly_y.max() + pad

        if force_even_dims:
            if (end_x - start_x) % 2:
                start_x = start_x - 1
            if (end_y - start_y) % 2:
                start_y = start_y - 1

        img_crop = img[start_y:end_y, start_x:end_x, :]
        mask_crop = mask[start_y:end_y, start_x:end_x]

        if asfloat:
            img_crop = uint8_to_float32(img_crop)
            if self.annotation_type == "csv":
                mask_crop = uint8_to_float32(mask_crop)

        return img_crop, mask_crop

    def select_lesion_ids(self):
        """Returns the IDs of non-empty selected lesions."""
        selected_lesion_ids = []
        for fitz_id in self.all_annotation_ids:
            sel_lesion_filepath = self.selected_lesion_filepath(fitz_id)
            mask = load_image(sel_lesion_filepath, mode="L")
            if np.max(mask) > 0:
                selected_lesion_ids.append(fitz_id)

        return selected_lesion_ids


class Background2d:
    """
    Base Class to add background scene to the renderings.
    """

    def __init__(
        self,
        dir_images: str,
        image_filenames: Optional[list] = None,
    ):
        self.dir_images = dir_images
        self.image_filenames = image_filenames

        if self.image_filenames is None:
            self.image_filenames = os.listdir(self.dir_images)

    def image_filepath(self, image_filename):
        return os.path.join(self.dir_images, image_filename)

    def random_image_filename(self):
        image_idx = np.random.randint(0, len(self.image_filenames))
        img_filename = self.image_filenames[image_idx]
        return img_filename

    def image(
        self,
        img_filename=None,
        img_size: Optional[tuple] = None,
        asfloat=False,
        resample=Image.BILINEAR,
        blur_radius: Optional[int] = None,
    ):
        if img_filename is None:
            img_filename = self.random_image_filename()

        img_filepath = self.image_filepath(img_filename)
        img = load_image(img_filepath, img_size=img_size, resample=resample)

        if blur_radius is not None:
            img = img.filter(ImageFilter.GaussianBlur(blur_radius))

        if asfloat:
            img = np.asarray(img, np.float32) / 255

        return img

    def hue_image(self, img_size=(512, 512)):
        back_img = np.ones(shape=(img_size + (3,)), dtype=np.float32)
        # Give background hues.
        back_img[:, :, 0] = np.round(random.uniform(0.75, 1), 2)
        back_img[:, :, 1] = np.round(random.uniform(0.0, 1), 2)
        back_img[:, :, 2] = np.round(random.uniform(0.0, 1), 2)

        return back_img


class NoGTDataset(ImageDataset):
    def __getitem__(self, idx: int):
        file_id = self.file_ids[idx]
        img = np.asarray(self.image(file_id))
        if self.spatial_transform:
            img_transformed = self.spatial_transform(image=img)
            img = img_transformed["image"]

        if self.image_augment:
            img_transformed = self.image_augment(image=img)
            img = img_transformed["image"]

        if self.image_preprocess:
            img_transformed = self.image_preprocess(image=img)
            img = img_transformed["image"]

        if self.totensor:
            img_transformed = self.totensor(image=img)
            img = img_transformed["image"]

        return self.name, file_id, img, []


class Ph2Dataset(ImageDataset):
    def file_ids(self):
        """Return a list of the unique file identifiers."""
        return [fname for fname in os.listdir(self.dir_images)]

    def image_filepath(self, image_id):
        return os.path.join(
            self.dir_images,
            image_id,
            image_id + "_Dermoscopic_Image",
            image_id + self.image_extension,
        )

    def target_filepath(self, image_id):
        return os.path.join(
            self.dir_targets,
            image_id,
            image_id + "_lesion",
            image_id + "_lesion" + self.image_extension,
        )

    def target(self, image_id, img_size: Optional[tuple] = None):
        target = load_image(self.target_filepath(image_id), img_size)
        # return np.asarray(target)[:,:,0]
        return target


class Fitz17KAnnotations(ImageDataset):
    def file_ids(self):
        # Custom directory structure
        self.annotators = os.listdir(self.dir_targets)
        self.folder_ids = []
        for ann in self.annotators:
            ann_filepath = os.path.join(self.dir_targets, ann)
            ann_file_ids = os.listdir(ann_filepath)
            for f_id in ann_file_ids:
                self.folder_ids.extend([os.path.join(ann, f_id)])

        self.file_ids_folders = {}
        for folder_file_id in self.folder_ids:
            file_id = folder_file_id.split("/")[1]
            self.file_ids_folders[file_id] = folder_file_id

        file_ids = [folder_file_id.split("/")[1] for folder_file_id in self.folder_ids]
        # IDs of the annotations with selected lesions that pass
        # the blending criteria. These IDs can be used for blending.
        self.annotation_ids = self.selected_lesion_ids(file_ids)
        return file_ids

    def selected_lesion_ids(self, file_ids=None):
        """Returns the IDs of non-empty selected lesions."""

        if file_ids is None:
            file_ids = self.file_ids

        selected_lesion_ids = []
        for fitz_id in file_ids:
            mask = self.selected_lesion(fitz_id)
            if can_blend_mask(mask):
                selected_lesion_ids.append(fitz_id)

        return selected_lesion_ids

    def lesions_filepath(self, image_id):
        folder_file_id = self.file_ids_folders[image_id]
        fpath = os.path.join(
            self.dir_targets, folder_file_id, "lesions" + self.target_extension
        )
        return fpath

    def nonskin_filepath(self, image_id):
        folder_file_id = self.file_ids_folders[image_id]
        return os.path.join(
            self.dir_targets, folder_file_id, "nonskin" + self.target_extension
        )

    def selected_lesion_filepath(self, image_id):
        folder_file_id = self.file_ids_folders[image_id]
        return os.path.join(
            self.dir_targets, folder_file_id, "selected_lesion" + self.target_extension
        )

    def mask(self, filepath, img_size=None):
        mask = load_image(
            filepath,
            img_size,
            mode="L",
            resample=Image.NEAREST,
        )

        mask = np.asarray(mask) / 255
        return mask.astype(np.float32)

    def lesions(self, image_id):
        return self.mask(self.lesions_filepath(image_id))

    def nonskin(self, image_id):
        return self.mask(self.nonskin_filepath(image_id))

    def selected_lesion(self, image_id, img_size=None):
        return self.mask(
            filepath=self.selected_lesion_filepath(image_id),
            img_size=img_size,
        )

    def target(self, image_id):
        lesions = np.asarray(self.lesions(image_id)) > 0
        nonskin = np.asarray(self.nonskin(image_id)) > 0
        healthy_skin = ~nonskin & ~lesions
        mask = np.zeros(shape=(lesions.shape[0], lesions.shape[1], 3), dtype=np.float32)
        mask[:, :, Target.LESION] = lesions * 1
        mask[:, :, Target.SKIN] = healthy_skin * 1
        mask[:, :, Target.NONSKIN] = nonskin * 1
        return mask  # Image.fromarray(mask)

    def box_crop_lesion(
        self,
        image_id: str,
        force_even_dims: bool = True,
        asfloat: bool = True,
    ):
        """Returns the image and mask cropped around the lesion.

        Args:
            image_id (str): The ID of image with a selected lesion.
            pad (int, optional): Amount to pad around the mask. Defaults to 8.

            force_even_dims (bool, optional):
                Force the returned cropped image to have even dimensions.
            asfloat (bool, optional): Return the cropped image as a float32,
                with the values scaled by 255. Else returns the original uint8.

        Returns:
            np.arrays: The cropped image and mask.
        """
        img = self.image(image_id)
        img = np.asarray(img)
        mask = self.selected_lesion(image_id)
        img_crop, mask_crop = box_crop_lesion(
            img=img,
            mask=mask,
            force_even_dims=force_even_dims,
            asfloat=asfloat,
        )

        return img_crop, mask_crop


class DermoFit(ImageDataset):
    def file_ids(self) -> List[str]:
        file_ids = [fname.split(".")[0] for fname in os.listdir(self.dir_images)]

        self.annotation_ids = []
        for file_id in file_ids:
            target = self.target(file_id)
            if can_blend_mask(np.asarray(target)[:, :, 0]):
                self.annotation_ids.append(file_id)

        return file_ids

    def binary_mask(self, image_id, img_size=None):
        filepath = self.target_filepath(image_id)
        mask = load_image(
            filepath,
            img_size,
            mode="L",
            resample=Image.NEAREST,
        )

        mask = np.asarray(mask) / 255
        return mask.astype(np.float32)

    def box_crop_lesion(
        self,
        image_id: str,
        force_even_dims: bool = True,
        asfloat: bool = True,
    ):
        img = self.image(image_id)
        img = np.asarray(img)
        mask = self.binary_mask(image_id)
        img_crop, mask_crop = box_crop_lesion(
            img=img,
            mask=mask,
            force_even_dims=force_even_dims,
            asfloat=asfloat,
        )
        return img_crop, mask_crop
