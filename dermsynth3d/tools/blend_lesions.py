from calendar import c
import os
import sys
import cv2
import yaml
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm, trange
from time import sleep

import torch
from scipy import ndimage
from pytorch3d.io import load_objs_as_meshes

from dermsynth3d.tools.renderer import MeshRendererPyTorch3D
from dermsynth3d.datasets.datasets import Fitz17KAnnotations
from dermsynth3d.deepblend.blend3d import Blended3d
from dermsynth3d.deepblend.blend import DeepTextureBlend3d, DeepImageBlend


class BlendLesions:
    """
    Handles the logic for blending the pasted lesions from stage 1.
    """

    def __init__(
        self,
        device,
        config,
    ):
        self.bodytex_dir = config["blending"]["bodytex_dir"]
        self.tex_dir = config["blending"]["tex_dir"]
        self.mesh_name = config["generate"]["mesh_name"]
        self.ext = config["blending"]["ext"]
        self.view_size = eval(config["generate"]["view_size"])
        self.num_iter = config["blending"]["num_iter"]
        self.lr = config["blending"]["lr"]
        self.device = device
        self.config = config
        self.SAVE_DIR = Path(config["generate"]["save_dir"].split("/")[0]) /"processed_textures"
        self.SAVE_DIR.mkdir(exist_ok=True, parents=True)
        self.SAVE_DIR = self.SAVE_DIR / self.mesh_name
        self.SAVE_DIR.mkdir(exist_ok=True, parents=True)
        print (f"Saving blended textures to {self.SAVE_DIR}")

        self.mesh_filename = os.path.join(
            self.bodytex_dir, self.mesh_name, "model_highres_0_normalized.obj"
        )

    def load_mesh(self):
        self.mesh = load_objs_as_meshes([self.mesh_filename], device=self.device)
        self.mesh_renderer = MeshRendererPyTorch3D(
            mesh=self.mesh,
            device=self.device,
            config=self.config,
        )

    def init_blenders(self):
        self.load_mesh()

        self.blend3d = Blended3d(
            mesh_filename=self.mesh_filename,
            device=self.device,
            dir_blended_textures=self.tex_dir,
            dir_nonskin_faces=None,
            extension=self.ext,
        )

        # Computes the deep blending loss
        self.deepblend = DeepImageBlend(gpu_id= self.device)

        # Wrapper for the blending process
        self.deepblend3d = DeepTextureBlend3d(
            self.blend3d,
            self.mesh_renderer,
            self.deepblend,
            self.device,
            self.view_size,
        )

    def optimize(self):
        print_every = 50

        self.optimizer.zero_grad()

        loss = self.deepblend3d.compute_loss_of_random_offset_view()
        loss.backward()

        step_idx = len(self.deepblend3d.deepblend.losses) - 1
        self.run[0] += 1
        return loss

    def blend_lesions(self):
        # Initialize the blenders
        self.init_blenders()

        self.optimizer = torch.optim.Adam(
            [self.deepblend3d.texture_image.requires_grad_()], lr=self.lr
        )

        # For each lesion, run a seperate optimization
        pbar1 = tqdm(total=self.deepblend3d.blended3d.lesion_params().shape[0], leave=True, desc="Blending Lesions")
        for _, params in self.deepblend3d.blended3d.lesion_params().iterrows():
            self.deepblend3d.set_params(params)
            self.run = [0]
            
            pbar = tqdm(total=self.num_iter, desc=f"Optimizing at face_idx ({int(self.deepblend3d.params.face_idx)})", leave=False, nrows=40)
            while self.run[0] <= self.num_iter:
                self.optimizer.step(self.optimize)
                # desc = f"L: {self.deepblend3d.deepblend.losses[-1]['loss']:.2f}, G: {self.deepblend3d.deepblend.losses[-1]['grad']:.2f}, S: {self.deepblend3d.deepblend.losses[-1]['style']:.2f}, C: {self.deepblend3d.deepblend.losses[-1]['content']:.2f}, TV: {self.deepblend3d.deepblend.losses[-1]['tv']:.2f}"
                desc = f"Loss: {self.deepblend3d.deepblend.losses[-1]['loss']:.3f}"
                pbar.set_postfix_str(desc, refresh=True)
                pbar.update(self.run[0] - pbar.n)
            pbar1.update(1)

        # Postprocess the blended textures
        merged_texture_np = self.deepblend3d.postprocess_blended_texture_image()

        # Save the final blended textures to disk.
        blended_filename = Path(self.SAVE_DIR) / f"model_highres_0_normalized_blended_{self.ext}.png"
        self.blend3d.save_blended_texture_image(merged_texture_np, print_filename=False, filename=blended_filename)

