import os
import json

from typing import Optional, Set
from PIL import Image

import numpy as np
import pandas as pd
from ast import literal_eval

from dermsynth3d.utils.image import load_image
from dermsynth3d.utils.tensor import pil_to_tensor


class Blended3d:
    """
    Loading and saving common files related to 3D blending.

    The code in the __init__ is specific to the 3DBodyTex.v1
    directory structure and would need to be changed for other datasets.
    """

    def __init__(
        self,
        mesh_filename: str,
        dir_blended_textures: str,
        dir_nonskin_faces: Optional[str] = None,
        dir_anatomy: Optional[str] = None,
        extension: str = "random_pytorch",
    ):
        self.dir_blended_textures = dir_blended_textures
        self.dir_nonskin_faces = None  # dir_nonskin_faces

        self.dir_anatomy = dir_anatomy

        # This assumes that `mesh_filename` is in the format of 3dBodyTex.v1
        self.dir_original_mesh = mesh_filename.split("model_highres_0_normalized.obj")[
            0
        ]

        split_filename = mesh_filename.split("/")
        self.subject_id = split_filename[-2]
        self.mesh_name = split_filename[-1]
        self.texture_name = self.mesh_name.split(".")[0]  # + '.png'
        self.extension = extension

        self.dir_subject = os.path.join(self.dir_blended_textures, self.subject_id)

        if not os.path.isdir(self.dir_subject):
            # Create the directory for the subject if it does not exist.
            os.mkdir(self.dir_subject)

    def vertices_to_anatomy(self):
        """
        Loads anatomy labels for each vertex.
        """
        anatomy_filename = os.path.join(
            self.dir_anatomy, self.subject_id, "vertslabels_scan.npy"
        )
        vertices_to_anatomy = np.load(anatomy_filename).squeeze()
        return vertices_to_anatomy

    def filepath_lesion_faces_uvs(self):
        print("***Warning: Function `filepath_lesion_faces_uvs(...)` is depreciated.")
        return os.path.join(
            self.dir_subject, "lesion_faces_uvs_" + self.extension + ".npy"
        )

    def save_lesion_faces_uvs(self, uvs_lesion):
        print("***Warning: Function `save_lesion_faces_uvs(...)` is depreciated.")
        np.save(self.filepath_lesion_faces_uvs(), uvs_lesion)

    def load_lesion_faces_uvs(self):
        print("***Warning: Function `load_lesion_faces_uvs(...)` is depreciated.")
        filepath = self.filepath_lesion_faces_uvs()
        uvs_lesion = np.load(filepath)
        return uvs_lesion

    def filepath_lesion_texture_mask(self):
        return os.path.join(self.dir_subject, "lesion_mask_" + self.extension + ".png")

    def save_lesion_texture_mask(self, lesion_texture_mask, print_filename=False):
        """Saves the lesion mask array as a PNG image.

        Image saved with a default filename and path.

        Args:
            lesion_texture_mask (np.ndarray): H x W numpy array
                representing the masked lesion of the image.
            print_filename (bool, optional): If True, prints the filepath.
        """
        filepath = self.filepath_lesion_texture_mask()
        Image.fromarray(lesion_texture_mask).save(filepath)
        if print_filename:
            print(filepath)

    def load_lesion_texture_mask(self):
        print("Depreciated: use `lesion_texture_mask()`")
        return self.lesion_texture_mask()

    def lesion_texture_mask(self, filepath=None, mode="L", astensor=False):
        if filepath is None:
            filepath = self.filepath_lesion_texture_mask()

        img = load_image(filepath, mode=mode)
        if astensor:
            img = pil_to_tensor(img)

        return img

    def texture_image(self, filepath=None, astensor=False):
        """Returns the original unmodified texture image."""
        if filepath is None:
            filepath = self.filepath_texture_image()
        img = load_image(filepath)
        if astensor:
            img = pil_to_tensor(img)

        return img

    def filepath_texture_image(self):
        fname = os.path.join(self.dir_original_mesh, "model_highres_0_normalized.png")
        return fname

    def filepath_blended_texture_image(self):
        return os.path.join(
            self.dir_subject, self.texture_name + "_" + self.extension + ".png"
        )

    def filepath_pasted_texture_image(self):
        return os.path.join(
            self.dir_subject, self.texture_name + "_pasted_" + self.extension + ".png"
        )

    def filepath_dilated_texture_image(self):
        return os.path.join(
            self.dir_subject, self.texture_name + "_dilated_" + self.extension + ".png"
        )

    def filepath_dilated_texture_mask(self):
        return os.path.join(
            self.dir_subject, "lesion_mask_dilated_" + self.extension + ".png"
        )

    def filepath_nonskin_texture_image(self):
        return os.path.join(
            self.dir_subject,
            "model_highres_0_normalized_mask.png",
        )

    def save_dilated_texture_mask(self, dilated_texture_mask, print_filepath=False):
        filepath = self.filepath_dilated_texture_mask()
        Image.fromarray(dilated_texture_mask).save(filepath)

        if print_filepath:
            print(filepath)

    def save_dilated_texture_image(self, dilated_texture_image, print_filepath=False):
        filepath = self.filepath_dilated_texture_image()
        Image.fromarray(dilated_texture_image).save(filepath)

        if print_filepath:
            print(filepath)

    def save_pasted_texture_image(self, pasted_texture_image, print_filepath=False):
        filepath = self.filepath_pasted_texture_image()
        Image.fromarray(pasted_texture_image).save(filepath)

        if print_filepath:
            print(filepath)

    def pasted_texture_image(self, astensor=False):
        img = load_image(self.filepath_pasted_texture_image())
        if astensor:
            img = pil_to_tensor(img)

        return img

    def save_blended_texture_image(self, blended_texture_image, print_filename=False):
        """Saves the texture image with a default filename and path.

        Args:
            blended_texture_image (np.ndarray): H x W x 3 numpy array
                representing the colored texture image.
            print_filename (bool, optional): If True, print the filepath
                to the saved image. Defaults to False.
        """
        filepath = self.filepath_blended_texture_image()
        Image.fromarray(blended_texture_image).save(filepath)

        if print_filename:
            print(filepath)

    def blended_texture_image(self, astensor=False):
        img = load_image(self.filepath_blended_texture_image())
        if astensor:
            img = pil_to_tensor(img)

        return img

    def dilated_texture_image(self, astensor=False):
        img = load_image(self.filepath_dilated_texture_image())
        if astensor:
            img = pil_to_tensor(img)

        return img

    def nonskin_texture_mask(self, astensor=False):
        img = load_image(self.filepath_nonskin_texture_image())
        if astensor:
            img = pil_to_tensor(img)

        return img

    def filepath_lesion_params(self):
        return os.path.join(
            self.dir_subject, "lesion_params_" + self.extension + ".csv"
        )

    def save_lesion_params(self, df, print_filepath=False):
        filepath = self.filepath_lesion_params()
        df.to_csv(filepath, index=False)
        if print_filepath:
            print(filepath)

    def lesion_params(self):
        df = pd.read_csv(self.filepath_lesion_params())
        if "look_at" in df:
            df["look_at"] = df["look_at"].apply(literal_eval)

        if "camera_pos" in df:
            df["camera_pos"] = df["camera_pos"].apply(literal_eval)

        if "normal" in df:
            df["normal"] = df["normal"].apply(literal_eval)

        return df

    def filepath_nonskin_face_indexes(self):
        print(
            "***Warning: Function `filepath_nonskin_face_indexes(...)` is depreciated."
        )
        return os.path.join(
            self.dir_nonskin_faces, self.subject_id + "_nonskin-face-indexes"
        )

    def nonskin_face_indexes(
        self,
        filepath: Optional[str] = None,
        ext=".npz",
    ) -> Set[int]:
        """Returns a set of mesh faces indexes that are of not skin (cloths, hair).

        These faces are manually labelled using Blender
        and the file `face_indexes.txt` must exist for the specific mesh.

        If no file exists, returns None.

        """
        if filepath is None:
            filepath = self.filepath_nonskin_face_indexes() + ext

        if ext == ".npz":
            loaded = np.load(filepath, allow_pickle=True)
            nonskin_face_indexes = loaded["nonskin_face_indexes"].item()
        else:
            # Text file.
            with open(filepath, "r") as wfile:
                nonskin_face_indexes = json.load(wfile)

        return set(nonskin_face_indexes)
