from calendar import c
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

        if self.run[0] % print_every == 0:
            step_idx = len(self.deepblend3d.deepblend.losses) - 1
            print(
                "run: {}, loss: {:4f}, grad: {:4f}, style: {:4f}, content: {:4f}, tv: {:4f},".format(
                    self.run,
                    self.deepblend3d.deepblend.losses[step_idx]["loss"],
                    self.deepblend3d.deepblend.losses[step_idx]["grad"],
                    self.deepblend3d.deepblend.losses[step_idx]["style"],
                    self.deepblend3d.deepblend.losses[step_idx]["content"],
                    self.deepblend3d.deepblend.losses[step_idx]["tv"],
                    # hist_loss.item(),
                )
            )
        self.run[0] += 1
        return loss

    def blend_lesions(self):
        # Initialize the blenders
        self.init_blenders()

        self.optimizer = torch.optim.Adam(
            [self.deepblend3d.texture_image.requires_grad_()], lr=self.lr
        )

        # For each lesion, run a seperate optimization
        for _, params in self.deepblend3d.blended3d.lesion_params().iterrows():
            self.deepblend3d.set_params(params)
            print(
                "Optimizing for face_idx = {}".format(
                    int(self.deepblend3d.params.face_idx)
                )
            )

            self.run = [0]

            while self.run[0] <= self.num_iter:
                self.optimizer.step(self.optimize)

        # Postprocess the blended textures
        merged_texture_np = self.deepblend3d.postprocess_blended_texture_image()

        # Save the final blended textures to disk.
        self.blend3d.save_blended_texture_image(merged_texture_np, print_filename=True)
