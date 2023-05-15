import os
from statistics import mode
import sys
import cv2
import yaml
import argparse

import numpy as np
import pandas as pd
import albumentations as A

import torch
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

from dermsynth3d.utils.utils import get_logger, yaml_loader
from dermsynth3d.models.model import SkinDeepLabV3
from dermsynth3d.datasets.datasets import Fitz17KAnnotations, ImageDataset, Ph2Dataset
from dermsynth3d.utils.colorconstancy import shade_of_gray_cc
from dermsynth3d.losses.metrics import compute_results, conf_mat_cells
from inference import inference_multitask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = yaml_loader("../configs/multitask.yaml")

# Load the model
multitask_model = SkinDeepLabV3(
    multi_head=config["infer"]["multi_head"], freeze_backbone=config["infer"]["freeze"]
)
multitask_model.load_state_dict(torch.load(config["infer"]["model_path"]), strict=False)

multitask_model = multitask_model.to(device)
multitask_model.eval()

img_size = eval(config["infer"]["img_size"])

# Assumes the model was pretrained using these values.
preprocess_input = A.Normalize(
    mean=eval(config["infer"]["mean"]), std=eval(config["infer"]["std"])
)
img_preprocess = A.Compose(
    [
        preprocess_input,
    ]
)

# To force a resize of the input image.
resize_func = A.Resize(
    height=img_size[0], width=img_size[1], interpolation=cv2.INTER_CUBIC
)
# Perform spatial augmentation on both the image and mask.
spatial_augment = A.Compose(
    [
        resize_func,
    ]
)
resize_aspect_smallest = A.augmentations.geometric.resize.SmallestMaxSize(
    max_size=img_size[0], always_apply=True
)
resize_aspect_longest = A.augmentations.geometric.resize.LongestMaxSize(
    max_size=img_size[0], always_apply=True
)

ph2_img_augment = A.Compose(
    [
        A.GaussianBlur(blur_limit=(9, 9), always_apply=True),
    ]
)

fitz_test_ds = Fitz17KAnnotations(
    dir_images=config["infer"]["fitz_imgs"],
    dir_targets=config["infer"]["fitz_annot"],
    image_extension=".jpg",
    target_extension=".png",
    spatial_transform=resize_aspect_smallest,
    image_augment=None,
    image_preprocess=img_preprocess,
    totensor=ToTensorV2(transpose_mask=True),
    color_constancy=shade_of_gray_cc,
)

fitz_test_dataloader = DataLoader(fitz_test_ds, batch_size=1, shuffle=False)

ph2_ds = Ph2Dataset(
    dir_images=config["infer"]["ph2_imgs"],
    dir_targets=None,
    name="ph2",
    image_extension=".bmp",
    target_extension=".bmp",
    image_augment=ph2_img_augment,
    spatial_transform=resize_aspect_longest,
    image_preprocess=img_preprocess,
    totensor=ToTensorV2(transpose_mask=True),
    color_constancy=shade_of_gray_cc,
)
ph2_dataloader = DataLoader(ph2_ds, batch_size=1, shuffle=False)

dermofit_img_augment = A.Compose(
    [
        A.GaussianBlur(blur_limit=(3, 3), always_apply=True),
    ]
)
dermofit_ds = ImageDataset(
    dir_images=config["infer"]["derm_imgs"],
    dir_targets=config["infer"]["derm_targets"],
    name="dermofit",
    image_extension=".png",
    target_extension=".png",
    image_augment=None,  # dermofit_img_augment,#dermofit_img_augment,
    spatial_transform=resize_aspect_smallest,  # resize_aspect_longest, #resize_aspect_smallest, #spatial_augment,
    image_preprocess=img_preprocess,
    totensor=ToTensorV2(transpose_mask=True),
    color_constancy=shade_of_gray_cc,
)
dermofit_dataloader = DataLoader(dermofit_ds, batch_size=1, shuffle=False)

pratheepan_ds = ImageDataset(
    dir_images=config["infer"]["prath_imgs"],
    dir_targets=config["infer"]["prath_tgts"],
    name="pratheepan",
    image_extension=".jpg",
    target_extension=".png",
    image_augment=None,
    spatial_transform=resize_aspect_smallest,
    image_preprocess=img_preprocess,
    totensor=ToTensorV2(transpose_mask=True),
    color_constancy=shade_of_gray_cc,
)
pratheepan_dataloader = DataLoader(pratheepan_ds, batch_size=1, shuffle=False)


def run_inference(mode, config):
    if mode == 1:
        fitz_test = inference_multitask(
            max_imgs=len(fitz_test_dataloader),
            model=multitask_model,
            dataloader=fitz_test_dataloader,
            device=device,
            save_to_disk=True,
            return_values=True,
            dir_anatomy_preds=config["infer"]["fitz_test_anatomy"],
            dir_save_images=config["infer"]["fitz_test_imgs"],
            dir_save_skin_preds=config["infer"]["fitz_test_skin"],
        )
        fitz_test_df = pd.DataFrame(
            compute_results(
                fitz_test_ds, config["infer"]["fitz_test_skin"], pred_ext=".png"
            )
        )

        logger.info(f"Fitzpatrick Dataset Size: {len(fitz_test_ds)}")

        logger.info("Skin condition:")
        logger.info(
            "{:.2f} \pm {:.2f}".format(
                fitz_test_df.lesion_ji.mean(),
                fitz_test_df.lesion_ji.std(),
            )
        )

        logger.info("Skin:")
        logger.info(
            "{:.2f} \pm {:.2f}".format(
                fitz_test_df.skin_ji.mean(),
                fitz_test_df.skin_ji.std(),
            )
        )

        logger.info("Non-Skin:")
        logger.info(
            "{:.2f} \pm {:.2f}".format(
                fitz_test_df.nonskin_ji.mean(),
                fitz_test_df.nonskin_ji.std(),
            )
        )

    elif mode == 2:
        ph2_preds = inference_multitask(
            max_imgs=len(ph2_dataloader),
            model=multitask_model,
            dataloader=ph2_dataloader,
            device=device,
            save_to_disk=True,
            return_values=False,
            dir_save_images=config["infer"]["ph2_test_images"],
            dir_save_targets=config["infer"]["ph2_test_targets"],
            dir_save_skin_preds=config["infer"]["ph2_test_skin"],
        )
        ph2_test_df = pd.DataFrame(
            compute_results(ph2_ds, config["infer"]["ph2_test_skin"], pred_ext=".bmp")
        )

        logger.info(f"Ph2 Dataset Size: {len(ph2_ds)}")

        logger.info("Skin condition:")
        logger.info(
            "{:.2f} \pm {:.2f}".format(
                ph2_test_df.lesion_ji.mean(),
                ph2_test_df.lesion_ji.std(),
            )
        )

    elif mode == 3:
        dermofit_preds = inference_multitask(
            max_imgs=len(dermofit_dataloader),
            model=multitask_model,
            dataloader=dermofit_dataloader,
            device=device,
            save_to_disk=True,
            return_values=False,
            dir_save_skin_preds=config["infer"]["derm_preds"],
        )
        dermofit_df = pd.DataFrame(
            compute_results(dermofit_ds, config["infer"]["derm_preds"], pred_ext=".png")
        )

        logger.info(f"DermoFit Dataset Size: {len(dermofit_ds)}")

        logger.info("Skin Conditions")
        logger.info(
            "{:.2f} \pm {:.2f}".format(
                dermofit_df.lesion_ji.mean(),
                dermofit_df.lesion_ji.std(),
            )
        )

    elif mode == 4:
        pratheepan_preds = inference_multitask(
            max_imgs=len(pratheepan_ds),
            model=multitask_model,
            dataloader=pratheepan_dataloader,
            device=device,
            save_to_disk=True,
            return_values=False,
            dir_anatomy_preds=config["infer"]["prath_preds_anatomy"],
            dir_save_skin_preds=config["infer"]["prath_preds_skin"],
        )
        pratheepan_df = pd.DataFrame(
            compute_results(
                pratheepan_ds, config["infer"]["prath_preds_skin"], pred_ext=".png"
            )
        )

        logger.info(f"Pratheepan Dataset Size: {len(pratheepan_ds)}")

        logger.info("Skin Condition:")
        logger.info(
            "{:.2f} \pm {:.2f}".format(
                pratheepan_df.skin_ji.mean(),
                pratheepan_df.skin_ji.std(),
            )
        )
        res = conf_mat_cells(pratheepan_ds, config["infer"]["prath_preds_skin"], ".png")
        tps = res["tps"]
        fps = res["fps"]
        fns = res["fns"]
        f1 = (2 * np.sum(tps)) / ((2 * np.sum(tps)) + np.sum(fps) + np.sum(fns))
        logger.info("F1 Score: ", f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        "-m",
        default=None,
        help="Select dataset to evaluate Multitask model. (1/2/3/4)",
    )
    args = parser.parse_args()

    if args.mode is not None:
        config["infer"]["mode"] = args.mode

    logger = get_logger(f'../logs/multitask_eval_{config["infer"]["mode"]}.log')

    run_inference(config["infer"]["mode"], config)
