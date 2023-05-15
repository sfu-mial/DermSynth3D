import os
import sys
import yaml
import random
import argparse
import logging
from PIL import Image
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../DermSynth3D")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../DermSynth3D/skin3d")))

from dermsynth3d.utils.utils import yaml_loader, get_logger
from dermsynth3d.datasets.datasets import (
    SynthDataset,
    ImageDataset,
    BinarySegementationDataset,
)
from dermsynth3d.losses.metrics import dice
from dermsynth3d.utils.image import float_img_to_uint8

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from dermsynth3d.losses.metrics import compute_results_segmentation

from scripts.inference import infer, evaluate, inference_segmentation

config = yaml_loader("./configs/train_mix.yaml")

# create directories for saving the checkpoints
os.makedirs(config["save_dir"], exist_ok=True)
if not os.path.exists(os.path.join(config["save_dir"], config["save_name"])):
    os.makedirs(os.path.join(config["save_dir"], config["save_name"]), exist_ok=True)

exp_name = f"{sys.argv[1][:-1]}_real{sys.argv[2]}_syn-mode{sys.argv[3]}_lr{config['train']['lr']}_rbr{config['train']['real_batch_ratio']}"

config["train"]["real_ratio"] = float(sys.argv[2])
config["train"]["mode"] = float(sys.argv[3])

SAVE_DIR = os.path.join(config["save_dir"], config["save_name"], exp_name)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

# Setup logger
LOG_DIR = os.path.join(SAVE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler(f"{LOG_DIR}/debug.log"))
# logger = logging.basicConfig(filename=LOG_DIR, level=logging.INFO)

# tensorboard
writer = SummaryWriter(LOG_DIR)

# Setup train/val/test paths
# TODO: Clean it further
import sys

root = f"./{sys.argv[1]}/syn/"
dir_tr_images = os.path.join(root, "train/images")
dir_tr_targets = os.path.join(root, "train/targets")

dir_real_images = os.path.join(config["real"], "train/images")
dir_real_targets = os.path.join(config["real"], "train/labels")

dir_val_images = os.path.join(config["real"], "test/images")
dir_val_targets = os.path.join(config["real"], "test/labels")

dir_test_images = os.path.join(config["real"], "validation/images")
dir_test_targets = os.path.join(config["real"], "validation/labels")

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setting up dataloaders/ augmentations
preprocess_input = A.Normalize(
    mean=eval(config["train"]["mean"]), std=eval(config["train"]["std"])
)
img_preprocess = A.Compose(
    [
        preprocess_input,
    ]
)
img_size = eval(config["train"]["img_size"])

# To force a resize of the input image.
resize_func = A.Resize(height=img_size[0], width=img_size[1])

# Perform spatial augmentation on both the image and mask.
spatial_augment = A.Compose(
    [
        A.HorizontalFlip(),
        A.RandomRotate90(),
        resize_func,
    ]
)
val_spatial_augment = A.Compose([resize_func])
min_v = config["train"]["min_v"]
max_v = config["train"]["max_v"]
img_augment = A.Compose(
    [
        A.ColorJitter(
            brightness=(min_v, max_v),
            contrast=(min_v, max_v),
            saturation=(min_v, max_v),
            hue=(-0.025, 0.025),
        ),
        A.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.75), always_apply=False),
        A.GaussianBlur(blur_limit=(3, 3)),
        A.ImageCompression(10, 100),
    ]
)

print("*" * 25)
print(f"Synthetic Dir Path: {dir_tr_images}")
print(f"Real Dir Path: {dir_real_images}")
print("*" * 25)

blend_ds = SynthDataset(
    dir_images=dir_tr_images,
    dir_targets=dir_tr_targets,
    name="synth_train",
    spatial_transform=spatial_augment,
    image_augment=img_augment,
    image_preprocess=preprocess_input,
    target_preprocess=None,
    target_extension=".npz",
    totensor=ToTensorV2(transpose_mask=True),
)

real_ds = BinarySegementationDataset(
    dir_images=dir_real_images,
    dir_targets=dir_real_targets,
    name="real_train",
    image_extension=".png",
    target_extension=".png",
    image_augment=None,
    spatial_transform=spatial_augment,
    image_preprocess=img_preprocess,
    totensor=ToTensorV2(transpose_mask=True),
)

real_val_ds = ImageDataset(
    dir_images=dir_val_images,
    dir_targets=dir_val_targets,
    name="real_val",
    image_extension=".png",
    target_extension=".png",
    image_augment=None,
    spatial_transform=val_spatial_augment,
    image_preprocess=img_preprocess,
    totensor=ToTensorV2(transpose_mask=True),
)

real_test_ds = ImageDataset(
    dir_images=dir_test_images,
    dir_targets=dir_test_targets,
    name="wound",
    image_extension=".png",
    target_extension=".png",
    image_augment=None,
    spatial_transform=val_spatial_augment,
    image_preprocess=img_preprocess,
    totensor=ToTensorV2(transpose_mask=True),
)

import math

# Selecting real images from the real data for training
num_real = math.ceil(len(real_ds) * config["train"]["real_ratio"])
sampled_file_ids = random.sample(real_ds.file_ids, num_real)
real_ds.file_ids = sampled_file_ids

# Select a fraction of real images for training
total_files = len(blend_ds)

sample_file_ids = random.sample(
    blend_ds.file_ids, math.ceil(total_files * config["train"]["mode"])
)
blend_ds.file_ids = sample_file_ids
# breakpoint()
logger.info(f"Total Synthetic Training data size: {total_files}")
logger.info(f"Synthetic Training Size: {len(blend_ds)}")
logger.info(f"Real Training Size: {len(real_ds)}")
logger.info(f"Real Validation Size: {len(real_val_ds)}")
logger.info(f"Real Test Size:  {len(real_test_ds)}")


# Setup dataloaders and  Force real samples in each batch
batch_size = config["train"]["batch_size"]
real_batch_size = int(config["train"]["real_batch_ratio"] * batch_size)
if len(blend_ds) == 0:
    blend_ds = real_ds
elif len(real_ds) == 0:
    real_ds = blend_ds

train_blend_dataloader = DataLoader(
    blend_ds, batch_size=(batch_size - real_batch_size), shuffle=True, drop_last=True
)

train_real_dataloader = DataLoader(
    real_ds, batch_size=real_batch_size, shuffle=True, drop_last=True
)

val_dataloader = DataLoader(
    real_val_ds, batch_size=config["test"]["batch_size"], shuffle=False, drop_last=True
)

test_dataloader = DataLoader(
    real_test_ds, batch_size=config["test"]["batch_size"], shuffle=False
)

# Setup Model
seg_model = deeplabv3_resnet50(pretrained=config["train"]["pretrain"])
n_classes = config["train"]["num_classes"]
seg_model.classifier = DeepLabHead(2048, n_classes)
seg_model = seg_model.to(device)
seg_model.train()

optimizer = torch.optim.Adam(
    seg_model.parameters(),
    lr=config["train"]["lr"],
    weight_decay=config["train"]["weight_decay"],
)

sigmoid = torch.nn.Sigmoid().to(device)
criterion = torch.nn.BCELoss().to(device)

train_losses = []
global step_idx


def train(config):
    step_idx = 0
    num_epochs = config["train"]["epochs"]

    start_epoch = 1

    max_val_iou = 0
    max_val_dice = 0
    max_test_iou = 0
    max_test_dice = 0

    logger.info("Starting Training.")

    # global step_idx

    for epoch in range(start_epoch, num_epochs + 1):
        seg_model.train()

        real_iter = iter(train_real_dataloader)
        total_train_loss = 0.0
        train_count = 0
        # breakpoint()
        for blend_batch in train_blend_dataloader:
            try:
                real_batch = next(real_iter)
            except:
                real_iter = iter(train_real_dataloader)
                real_batch = next(real_iter)
            # Combine synthetic and real. Index 2 contains the images.
            images = torch.cat((blend_batch[2], real_batch[2]), dim=0)
            # Combine targets. Index 3 contains the targets.
            # The 0:1 gets the first index of the targets, which should correspond to the lesion mask.
            segs = torch.cat(
                (
                    blend_batch[3][:, 0:n_classes, :, :],
                    real_batch[3][:, 0:n_classes, :, :],
                ),
                dim=0,
            )
            # breakpoint()

            optimizer.zero_grad()

            images = images.to(device)
            segs = segs.to(device).float()
            out = seg_model(images)["out"]
            preds = sigmoid(out)

            loss = criterion(preds, segs)  # [:, :n_classes, :, :])
            if loss < 0:
                raise ValueError("Negative loss")

            train_losses.append([step_idx, loss.item()])
            writer.add_scalar("train/loss", loss.item(), step_idx)

            step_idx += 1  # images.size(0)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_count += 1

            if step_idx % config["val"]["log_every"] == 0:
                # for logging only
                logger.info(
                    f"[TRAIN] Epoch: {epoch}/{num_epochs} | Step: {step_idx} | Loss: {(total_train_loss/train_count):.5f}"
                )

            if step_idx % config["val"]["val_every"] == 0:
                logger.info(
                    f"[TRAIN] Epoch: {epoch}/{num_epochs} | Step: {step_idx} | Loss: {(total_train_loss/train_count):.5f}"
                )

                seg_model.eval()

                val_score = evaluate(seg_model, val_dataloader, device, real_val_ds)
                val_iou = val_score.iou.mean()
                val_dice = val_score.dice.mean()

                writer.add_scalar("val/dice", val_dice, step_idx)
                writer.add_scalar("val/iou", val_iou, step_idx)

                if val_dice > max_val_dice:
                    # path for saving the preds
                    dir_seg_preds_wounds_test = os.path.join(
                        SAVE_DIR,
                        f"epoch{epoch}-step{step_idx}-preds-val_dice{val_dice:.5f}",
                    )
                    test_score = evaluate(
                        seg_model,
                        test_dataloader,
                        device,
                        real_test_ds,
                        save_to_disk=True,
                        save_dir=dir_seg_preds_wounds_test,
                        writer=writer,
                    )

                    max_val_dice = val_dice
                    max_test_dice = max(test_score.dice.mean(), max_test_dice)
                    max_test_iou = max(max_test_iou, test_score.iou.mean())

                    writer.add_scalar("test/dice", max_test_dice, step_idx)
                    writer.add_scalar("test/iou", max_test_iou, step_idx)

                    logger.info(
                        f"Saving model with val dice:{max_val_dice:.4f} and test dice: {max_test_dice:0.4f} at epoch {epoch} iteration: {step_idx}"
                    )

                    os.makedirs(f"{SAVE_DIR}/models/", exist_ok=True)
                    PATH = (
                        SAVE_DIR
                        + f"/models/epoch{epoch}_step{step_idx}_val_dice{max_val_dice:.4f}_test_dice{max_test_dice:.4f}.pt"
                    )

                    torch.save(deepcopy(seg_model.state_dict()), PATH)
                    logger.info(f"Saving at --> {PATH}")

                    logger.info("*********************************")
                    logger.info(f"[Test with best dice model] iou: {max_test_iou:.4f}")
                    logger.info(
                        f"[Test with best dice model] dice: {max_test_dice:.4f}"
                    )
                    logger.info("*********************************")

                if val_iou > max_val_iou:
                    test_score = evaluate(
                        seg_model, test_dataloader, device, real_test_ds
                    )
                    max_val_iou = val_iou
                    max_test_dice = max(test_score.dice.mean(), max_test_dice)
                    max_test_iou = max(max_test_iou, test_score.iou.mean())

                    logger.info(
                        f"Saving model with val iou:{max_val_iou:.4f} and test iou: {max_test_iou:0.4f} at epoch {epoch} iteration: {step_idx}"
                    )
                    os.makedirs(f"{SAVE_DIR}/models/", exist_ok=True)
                    PATH = (
                        SAVE_DIR
                        + f"/models/epoch{epoch}_step{step_idx}_val_iou{max_val_iou:.4f}_test_iou{max_test_iou:.4f}.pt"
                    )

                    torch.save(deepcopy(seg_model.state_dict()), PATH)
                    logger.info(f"Saving at --> {PATH}")

                    logger.info("*********************************")
                    logger.info(f"[Test with best iou model] iou: {max_test_iou:.4f}")
                    logger.info(f"[Test with best iou model] dice: {max_test_dice:.4f}")
                    logger.info("*********************************")

                seg_model.train()

    logger.info("Training Finished.")


if __name__ == "__main__":
    train(config)
