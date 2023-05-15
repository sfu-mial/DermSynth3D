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
from dermsynth3d.deepblend.blend import PasteTextureImage
from dermsynth3d.deepblend.blend3d import Blended3d
from dermsynth3d.utils.textures import UVViewMapper
from dermsynth3d.utils.colorconstancy import shade_of_gray_cc


class SelectAndPaste:
    """
    Handles the logic of selecting the ideal location
    for pasting/blending the lesion.
    """

    def __init__(
        self,
        device,
        config,
    ):
        self.bodytex_dir = config["blending"]["bodytex_dir"]
        self.dir_fitzk_images = config["blending"]["fitz_dir"]
        self.annot_dir = config["blending"]["annot_dir"]
        self.tex_dir = config["blending"]["tex_dir"]
        self.mesh_name = config["generate"]["mesh_name"]
        self.view_size = eval(config["generate"]["view_size"])
        self.ext = config["blending"]["ext"]
        self.num_paste = config["blending"]["num_paste"]
        self.device = device
        self.config = config

        self.fitz_ds = Fitz17KAnnotations(
            dir_images=self.dir_fitzk_images,
            dir_targets=self.annot_dir,
            target_extension=config["extension"]["target_extension"],
            image_extension=config["extension"]["image_extension"],
            color_constancy=shade_of_gray_cc,
        )
        os.makedirs(self.tex_dir, exist_ok=True)

        # Assumes 3DBodyTex v1. file naming convension
        self.mesh_filename = os.path.join(
            self.bodytex_dir, self.mesh_name, "model_highres_0_normalized.obj"
        )

        self.blended3d = Blended3d(
            mesh_filename=self.mesh_filename,
            dir_blended_textures=self.tex_dir,
            dir_nonskin_faces=None,
            extension=self.ext,
        )

        # Load the non-skin and original texture map
        self.nonskin_texture_mask_tensor = self.blended3d.nonskin_texture_mask(
            astensor=True
        )
        self.original_texture_image_tensor = self.blended3d.texture_image(astensor=True)

    def load_mesh(self):
        # Load the mesh
        self.mesh = load_objs_as_meshes([self.mesh_filename], device=self.device)
        self.mesh_renderer = MeshRendererPyTorch3D(
            mesh=self.mesh,
            device=self.device,
            config=self.config,
        )
        self.faces_is_skin = self.mesh_renderer.faces_is_skin(
            self.nonskin_texture_mask_tensor
        )

    def load_paster(self):
        self.paster = PasteTextureImage(
            self.original_texture_image_tensor,
            self.nonskin_texture_mask_tensor,
            self.mesh_renderer,
            self.view_size,
        )

        # Initialize the textures and masks to paste the lesions
        self.paster.initialize_pasted_masks_and_textures()

    def paste_on_locations(self):
        # Initialize mesh and paster
        self.load_mesh()
        self.load_paster()

        # Constrain the random faces to be ones of skin (not cloths).
        self.rand_skin_face_indexes = np.random.permutation(
            np.where(self.faces_is_skin)[0]
        )

        # Stores the parameters of the pasted texture image.
        self.selected_params = []

        # Id to start the lesion masks.
        lesion_mask_id = 1
        # Start from the first randomly permuted face.
        face_iter = 0

        # Repeat until paste `n_paste` number of lesions.
        while lesion_mask_id <= self.num_paste:
            # Select a random face.
            face_idx = self.rand_skin_face_indexes[face_iter]
            face_iter += 1

            # Set the renderer to view the face.
            self.paster.set_renderer_view(face_idx)

            # Select a random lesion.
            fitz_id = np.random.permutation(self.fitz_ds.annotation_ids)[0]
            lesion_img, lesion_seg = self.fitz_ds.box_crop_lesion(
                fitz_id,
                force_even_dims=True,
                asfloat=True,
            )

            # Set the lesion and mask.
            self.paster.init_target_lesion(lesion_img, lesion_seg)
            # Compute the max change in depth for the lesion location.
            max_depth_diff = self.paster.lesion_max_depth_diff()

            if max_depth_diff > 0.02:
                # If max change exceeds a threshold, skip it.
                print("Skipping {} due to high depth change".format(face_idx))
                continue

            accepted = self.paster.paste_masks_and_texture_images(lesion_mask_id)
            if not accepted:
                print("Skipping {} as overlaps with existing lesion.".format(face_idx))
                continue

            self.selected_params.append(
                {
                    "face_idx": face_idx,
                    "normal_weight": self.paster.mesh_renderer.params["normal_weight"],
                    "max_depth_diff": max_depth_diff,
                    "lesion_mask_id": lesion_mask_id,
                    "fitz_id": fitz_id,
                }
            )

            print("Accepted lesion_mask_id={}".format(lesion_mask_id))
            lesion_mask_id += 1

        self.save_textures()

    def save_textures(self):
        # Save the pasted texture image.
        self.blended3d.save_pasted_texture_image(
            self.paster.pasted_texture, print_filepath=True
        )
        self.blended3d.save_dilated_texture_image(
            self.paster.pasted_dilated_texture, print_filepath=True
        )

        # Save the lesion mask for the texture image.
        self.blended3d.save_lesion_texture_mask(
            self.paster.lesion_mask.squeeze(), print_filename=True
        )
        self.blended3d.save_dilated_texture_mask(
            self.paster.lesion_dilated_mask.squeeze(), print_filepath=True
        )

        # Save the location parameters of the selected views.
        final_params_df = pd.DataFrame(self.selected_params)
        self.blended3d.save_lesion_params(final_params_df, print_filepath=True)
