"""
This script is used to prepare the data for training models with synthetic data.
It will create a train and val folder with images and targets subfolders.
"""

from glob import glob
import os
import random

subjects = glob("./out/blend_lesions/*")
train_dir = "train"
val_dir = "val"

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    os.makedirs(os.path.join(train_dir, "images"))
    os.makedirs(os.path.join(train_dir, "targets"))
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
    os.makedirs(os.path.join(val_dir, "images"))
    os.makedirs(os.path.join(val_dir, "targets"))

train_subs = random.sample(subjects, int(len(subjects) * 0.9))
val_subs = set(subjects) - set(train_subs)

train_imgs = []
val_imgs = []

for sub in train_subs:
    files = random.sample(glob(os.path.join(sub, "images/*.png")), 400)
    train_imgs.extend(files)

for sub in val_subs:
    files = random.sample(glob(os.path.join(sub, "images/*.png")), 400)
    val_imgs.extend(files)

for f in train_imgs:
    sp = f.split("/")
    os.system(f"cp {f} {train_dir}/images/")
    os.system(
        f"cp ./out/blend_lesions/{sp[2]}/targets/{sp[-1][:-4]}.npz {train_dir}/targets"
    )

for f in val_imgs:
    sp = f.split("/")
    os.system(f"cp {f} {val_dir}/images/")
    os.system(
        f"cp ./out/blend_lesions/{sp[2]}/targets/{sp[-1][:-4]}.npz {val_dir}/targets"
    )
