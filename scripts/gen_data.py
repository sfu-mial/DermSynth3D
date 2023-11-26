import os
import sys
import cv2
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from scipy import ndimage
from pprint import pprint

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "skin3d"))
)

from dermsynth3d import BlendLesions, Generate2DViews, SelectAndPaste

from dermsynth3d.utils.utils import yaml_loader

# Setup device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
# combine all yamls into one
default = yaml_loader("configs/default.yaml")
main = yaml_loader("configs/blend.yaml")
main.update(default)
renderer = yaml_loader("configs/renderer.yaml")
main.update(renderer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh_name", "-m", default=None, type=str, help="Name of mesh"
    )
    parser.add_argument(
        "--lr", default=None, type=float, help="learning rate for optimization"
    )
    parser.add_argument(
        "--percent_skin",
        "-ps",
        default=0.1,
        type=float,
        help="skin threshold for saving the view",
    )
    parser.add_argument(
        "--num_iter",
        "-i",
        default=None,
        type=int,
        help="number of iterations for blending",
    )
    parser.add_argument(
        "--num_views", "-v", default=None, type=int, help="number of views to generate"
    )
    parser.add_argument(
        "--save_dir",
        "-s",
        default=None,
        type=str,
        help="path to save the generated views",
    )
    parser.add_argument(
        "--num_paste",
        "-n",
        default=None,
        type=int,
        help="number of lesions to paste per mesh",
    )
    parser.add_argument(
        "--paste", action="store_true", help="whether to force pasting or not"
    )

    args = parser.parse_args()
    args = vars(args)
    for key in args:
        if args[key] is not None:
            if key in main["blending"]:
                main["blending"][key] = args[key]
            if key in main["generate"]:
                main["generate"][key] = args[key]
    pprint(main)
    locations = SelectAndPaste(config=main, device=device)
    locations.paste_on_locations()
    blender = BlendLesions(config=main, device=device)
    blender.blend_lesions()
    renderer = Generate2DViews(config=main, device=device)
    renderer.synthesize_views()
