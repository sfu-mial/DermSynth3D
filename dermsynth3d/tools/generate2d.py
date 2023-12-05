import os
import yaml
import random
import trimesh
import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path
from tqdm import tqdm, trange

import torch
import pytorch3d

from scipy.stats import skewnorm
from sklearn.preprocessing import normalize

from pytorch3d.io import load_objs_as_meshes

from dermsynth3d.utils.utils import random_bound, make_masks
from dermsynth3d.tools.synthesize import Synthesize2D
from dermsynth3d.datasets.synth_dataset import SynthesizeDataset
from dermsynth3d.tools.renderer import (
    MeshRendererPyTorch3D,
    camera_pos_from_normal,
)
from dermsynth3d.deepblend.blend3d import Blended3d
from dermsynth3d.utils.channels import Target
from dermsynth3d.utils.colorconstancy import shade_of_gray_cc
from dermsynth3d.datasets.datasets import Fitz17KAnnotations, Background2d
from skin3d.skin3d.bodytex import BodyTexDataset


class Generate2DHelper:
    """
    Generates the 2D views from the 3D mesh.
    Handles the logic to determine appropriate camera and light positions.
    Handles the logic to paste the lesion.
    """

    def __init__(
        self,
        mesh_filename: str,
        dir_blended_textures: str,
        dir_anatomy: str,
        fitz_ds,
        background_ds,
        device,
        blended_file_ext,
        config,
        debug=True,
        bodytex=None,
        percent_skin=0.30,
        is_blended: bool = True,  # If True, use blended textures. Else use pasted textures.
    ):
        # If True, prints internal messages
        # if image does not meet criteria.
        self.debug = debug
        self.bodytex = bodytex
        self.texture_nevi_mask = None
        self.percent_skin = percent_skin
        self.config = config
        self.device = device

        # Load the mesh in pytorch3d.
        mesh = load_objs_as_meshes([mesh_filename], device=self.device)
        self.mesh_renderer = MeshRendererPyTorch3D(
            mesh,
            self.device,
            config=self.config,
        )

        # Load the mesh using trimesh.
        self.mesh_tri = trimesh.load(mesh_filename, process=False, maintain_order=True)

        self.blended3d = Blended3d(
            mesh_filename=mesh_filename,
            device=self.device,
            dir_blended_textures=dir_blended_textures,
            dir_anatomy=dir_anatomy,
            extension=blended_file_ext,
        )

        # Load the texture image and corresponding mask of the blended lesion.
        if is_blended:
            self.blended_texture_image = self.blended3d.blended_texture_image(
                astensor=True
            ).to(self.device)
        else:
            self.blended_texture_image = self.blended3d.pasted_texture_image(
                astensor=True
            ).to(self.device)

        self.texture_lesion_mask = self.blended3d.lesion_texture_mask(astensor=True).to(self.device)
        self.nonskin_texture_mask_tensor = self.blended3d.nonskin_texture_mask(
            astensor=True
        ).to(self.device)

        self.vertices_to_anatomy = self.blended3d.vertices_to_anatomy()

        self.fitz_ds = fitz_ds
        self.background_ds = background_ds

        if self.bodytex is not None:
            scan_id = self.blended3d.subject_id[:3]
            # if a bodytex annotation from skin3d does not exist
            # then cannot continue to get the nevi annotations in skin3d
            self.nevi_exists = os.path.exists(self.bodytex.annotation_filepath(scan_id))
            if not self.nevi_exists:
                raise ValueError("Missing Bodytex annotations for creating GT Nevi")

            self.texture_nevi_mask = np.zeros(
                shape=self.texture_lesion_mask.shape, dtype=np.float32
            )
            ann_df = self.bodytex.annotation(scan_id, annotator=None)
            for _, row in ann_df.iterrows():
                self.texture_nevi_mask[
                    row.y : row.y2,
                    row.x : row.x2,
                ] = 255

            self.texture_nevi_mask = torch.tensor(
                self.texture_nevi_mask / 255, dtype=torch.float32
            ).to(device)

        self.ray = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh_tri)

        self.perspective_correct = False

        self.face_idx = None
        self.normal_weight = None
        self.view_size = []
        self.look_at = []
        self.camera_pos = []
        self.light_pos = []
        self.ambient = []
        self.specular = []
        self.diffuse = []
        self.znear = None
        self.paste_lesion_id = None
        self.background_id = None
        self.shininess = None

    def get_params(self):
        params = {
            "bodytex_id": self.blended3d.subject_id,
            "face_idx": self.face_idx,
            "normal_weight": self.normal_weight,
            "view_size": self.view_size,
            "look_at": list(self.look_at),
            "ambient": list(self.ambient),
            "specular": list(self.specular),
            "diffuse": list(self.diffuse),
            "camera_pos": list(self.camera_pos),
            "light_pos": list(self.light_pos),
            "shininess": list(self.shininess),
            "znear": self.znear,
            "paste_lesion_id": self.paste_lesion_id,
            "background_id": self.background_id,
            "background_blur_radius": self.background_blur_radius,
        }
        return params

    def render_image_and_target(
        self,
        paste_lesion: bool = True,
        min_fraction_lesion: float = 0.0,
    ):
        """Returns the rendered image and targets.

        Returns (None, None) if the rendered image does not pass certain criteria.

        Args:
            paste_lesion (bool, optional): If true, paste the lesion onto the skin.
            min_fraction_lesion (float, optional): Fraction between 0 and 1 of the image
                that the lesion must occupy for the image/target to be returned.
                Defaults to 0, which means the lesion is not required to be in the image.

        Returns:
            _type_: _description_
        """

        paste_img, paste_mask = self.composite_image(
            paste_lesion=paste_lesion,
            min_fraction_lesion=min_fraction_lesion,
        )

        if paste_img is None:
            return None, None

        anatomy_image = self.render_anatomy_image()
        depth_view = self.mesh_renderer.depth_view()
        nevi_view = np.zeros_like(depth_view)
        if self.bodytex is not None and self.nevi_exists:
            nevi_view = self.render_nevi_square_mask()

        n_target_channels = 5
        if self.bodytex is not None and self.nevi_exists:
            n_target_channels = 6

        target = np.zeros(
            shape=(paste_img.shape[0], paste_img.shape[1], n_target_channels),
            dtype=np.float32,
        )
        target[:, :, :3] = paste_mask
        target[:, :, Target.ANATOMY] = anatomy_image
        target[:, :, Target.DEPTH] = depth_view

        if self.bodytex is not None:
            target[:, :, Target.NEVI] = nevi_view

        return paste_img, target

    def render_anatomy_image(self):
        anatomy_image = self.mesh_renderer.anatomy_image(self.vertices_to_anatomy)
        return anatomy_image

    def composite_image(self, paste_lesion=True, soft_mask=True, min_fraction_lesion=0):
        if (min_fraction_lesion < 0) or (min_fraction_lesion > 1):
            raise ValueError("`min_fraction_lesion` must be between 0 and 1.")

        back_img, background_id = self.background_image(view_size = self.view_size)
        self.background_id = background_id

        skin_mask = self.render_skin_mask()
        n_pixels = skin_mask.shape[0] * skin_mask.shape[1]

        # Fraction of the image that must contain skin.
        if skin_mask.sum() < (n_pixels * self.percent_skin):
            if self.debug:
                print("***Not enough skin.")
            return None, None

        lesion_mask = self.render_lesion_mask()

        if paste_lesion:
            # Select a random lesion.
            lesion_id = np.random.permutation(self.fitz_ds.annotation_ids)[0]
            lesion_crop_orig, seg_crop_orig = self.fitz_ds.box_crop_lesion(
                lesion_id,
                force_even_dims=True,
                asfloat=False,
            )

            synth2d = Synthesize2D(
                view2d=self.render_view(),
                body_mask=self.render_body_mask(),
                skin_mask=skin_mask,
                lesion_mask=lesion_mask,
                background_img=back_img,
                paste_lesion_img=lesion_crop_orig,
                paste_lesion_mask=seg_crop_orig,
                soft_blend=soft_mask,
            )
            min_scale, max_scale = synth2d.min_max_scale_foreground()
            paste_img, paste_mask = synth2d.random_paste_foreground_with_retry(
                min_scale, max_scale
            )
            self.paste_lesion_id = lesion_id

        else:
            synth2d = Synthesize2D(
                view2d=self.render_view(),
                body_mask=self.render_body_mask(),
                skin_mask=skin_mask,
                lesion_mask=lesion_mask,
                background_img=back_img,
                paste_lesion_img=None,
                paste_lesion_mask=None,
                soft_blend=soft_mask,
            )
            paste_img = synth2d.pasted_back_image
            paste_mask = make_masks(lesion_mask, skin_mask)
            self.paste_lesion_id = None

        if paste_mask[:, :, Target.LESION].sum() < (n_pixels * min_fraction_lesion):
            if self.debug:
                print("***Not enough lesion.")
            return None, None

        return paste_img, paste_mask

    def background_image(self, view_size=(512, 512)):
        background_id = self.background_ds.random_image_filename()

        blur_radius = None
        if self.background_blur_radius > 0:
            blur_radius = self.background_blur_radius

        back_img = self.background_ds.image(
            img_filename=background_id,
            asfloat=True,
            img_size=view_size,
            blur_radius=blur_radius,
        )
        return back_img, background_id

    def random_face_idx(self):
        return np.random.randint(0, self.mesh_renderer.center_face_vertices.shape[0])

    def random_light_pos(self, camera_pos, look_at):
        signed_pos = np.sign(camera_pos - look_at) * 2
        light_offset = np.asarray([random_bound(0, s) for s in signed_pos])
        light_pos = light_offset + camera_pos
        return light_pos

    def random_light_color(self, lower, upper):
        return [np.round(random_bound(lower, upper), 2)] * 3

    def random_shininess(self, lower, uppper):
        return [np.round(random_bound(lower, uppper), 2)]

    def randomize_parameters(
        self,
        config,
        face_idx=None,
        sample_mode="sample_surface",
        look_at=None,
        look_at_normal=None,
        view_size=(512, 512),
        surface_offset_bounds=(0.1, 1.3),
        ambient_bounds=(0.3, 0.99),
        specular_bounds=(0, 0.1),
        diffuse_bounds=(0.3, 0.99),
        mat_diffuse_bounds=(0.3, 0.99),
        mat_specular_bounds=(0.0, 0.05),
        znear=0.01,
        light_pos=None,
        shininess=(30, 60),
        sphere_pos=False,
        elev_bounds=(0, 180),
        azim_bounds=(-90, 90),
        background_blur_radius_bounds=(0, 3),
    ):
        if config is not None:
            view_size = eval(config["generate"]["view_size"])
            surface_offset_bounds = eval(
                config["generate"]["random"]["surface_offset_bounds"]
            )
            ambient_bounds = eval(config["generate"]["random"]["ambient_bounds"])
            specular_bounds = eval(config["generate"]["random"]["specular_bounds"])
            diffuse_bounds = eval(config["generate"]["random"]["diffuse_bounds"])
            mat_diffuse_bounds = eval(
                config["generate"]["random"]["mat_diffuse_bounds"]
            )
            mat_specular_bounds = eval(
                config["generate"]["random"]["mat_specular_bounds"]
            )
            znear = config["generate"]["random"]["znear"]
            light_pos = eval(config["generate"]["random"]["light_pos"])
            shininess = eval(config["generate"]["random"]["shininess"])
            sphere_pos = config["generate"]["random"]["sphere_pos"]
            elev_bounds = eval(config["generate"]["random"]["elev_bounds"])
            azim_bounds = eval(config["generate"]["random"]["azim_bounds"])
            background_blur_radius_bounds = eval(
                config["generate"]["random"]["background_blur_radius_bounds"]
            )

        # surface_offset_skew = None,

        self.elev = random_bound(*elev_bounds)
        self.azim = random_bound(*azim_bounds)

        self.background_blur_radius = random.randint(
            background_blur_radius_bounds[0], background_blur_radius_bounds[1]
        )

        self.view_size = view_size
        self.znear = znear
        # Determine how far to place the camera from the face.
        self.normal_weight = random_bound(*surface_offset_bounds)

        self.face_idx = face_idx
        self.look_at = look_at
        self.look_at_normal = look_at_normal

        if sample_mode == "sample_face":
            if self.face_idx is None:
                self.face_idx = self.random_face_idx()

            # Get 3D coordinates for the camera = `camera_pos`
            # Get 3D coordinates for where the camera is looking at = `look_at`
            (
                self.camera_pos,
                self.look_at,
            ) = self.mesh_renderer.camera_parameters_offset_face(
                self.face_idx, normal_weight=self.normal_weight
            )
        elif sample_mode == "sample_surface":
            coords, normals = pytorch3d.ops.sample_points_from_meshes(
                meshes=self.mesh_renderer.mesh,
                num_samples=1,
                return_normals=True,
                return_textures=False,
            )

            if self.look_at is None:
                self.look_at = coords.cpu().detach().numpy().squeeze()
            if look_at_normal is None:
                self.look_at_normal = normals.cpu().detach().numpy().squeeze()

            self.camera_pos = camera_pos_from_normal(
                look_at=self.look_at,
                normal=self.look_at_normal,
                normal_weight=self.normal_weight,
            )
        else:
            raise NotImplementedError(
                "Error: `{}` is not a valid option for `sample_mode`.".format(
                    sample_mode
                )
            )

        # Check if the camera is placed inside a mesh.
        camera_signed_d = trimesh.proximity.signed_distance(
            self.mesh_tri, [self.camera_pos]
        )
        if camera_signed_d >= 0:
            if self.debug:
                print("Camera inside mesh. Skip.")
            return False

        self.light_pos = light_pos
        if self.light_pos is None or self.light_pos == "None":
            self.light_pos = self.random_light_pos(self.camera_pos, self.look_at)

        normal = normalize([self.camera_pos - self.look_at])[0]
        if self.mesh_intersects(self.camera_pos + normal * 0.01, self.light_pos):
            # if self.mesh_intersects(self.look_at + self.normal_weight*0.01, self.light_pos):
            if self.debug:
                print("Lighting blocked by mesh. Setting lighting to camera pos.")
            self.light_pos = self.camera_pos

        self.ambient = self.random_light_color(*ambient_bounds)
        self.specular = self.random_light_color(*specular_bounds)
        self.diffuse = self.random_light_color(*diffuse_bounds)
        self.mat_diffuse = self.random_light_color(*mat_diffuse_bounds)
        self.mat_specular = self.random_light_color(*mat_specular_bounds)
        self.shininess = self.random_shininess(*shininess)

        self.initialize_from_params(sphere_pos)

        return True

    def initialize_from_params(self, sphere_pos):
        camera_pos = self.camera_pos
        dist = None
        elev = None
        azim = None
        up = None
        if sphere_pos:
            camera_pos = None
            dist = self.normal_weight
            elev = self.elev
            azim = self.azim
            up = np.asarray(
                [
                    normalize([self.camera_pos - self.look_at])[0],
                ]
            )

        self.mesh_renderer.precompute_view_parameters(
            view_size=self.view_size,
            at=self.look_at,
            dist=dist,
            elev=elev,
            azim=azim,
            camera_pos=camera_pos,
            znear=float(self.znear),
            perspective_correct=self.perspective_correct,
            up=up,
        )

        self.set_image_parameters()

        self.mesh_renderer.initialize_renderer()
        self.mesh_renderer.compute_fragments()

    def mesh_intersects(self, source_point, target_point):
        is_intersect = self.ray.intersects_any(
            [source_point], [np.asarray(target_point) - np.asarray(source_point)]
        )
        return is_intersect

    def render_view(self):
        # self.initialize_from_params()
        self.mesh_renderer.set_texture_image(self.blended_texture_image)
        view2d = self.mesh_renderer.render_view(asnumpy=True, asRGB=True)
        return view2d

    def render_lesion_mask(self):
        self.mesh_renderer.set_texture_image(self.texture_lesion_mask[:, :, np.newaxis])
        self.set_mask_parameters()
        mask2d = self.mesh_renderer.render_view(asnumpy=True, asRGB=True)
        lesion_mask = self.mesh_renderer.lesion_mask(
            mask2d[:, :, 0], lesion_mask_id=None
        )

        # Set the illumination back to the earlier settings.
        self.set_image_parameters()
        self.mesh_renderer.initialize_renderer()
        return lesion_mask

    def render_nevi_square_mask(self):
        if self.bodytex is None:
            raise ValueError("Error: cannot render nevi masks without `self.bodytex`.")

        self.mesh_renderer.set_texture_image(self.texture_nevi_mask[:, :, np.newaxis])
        self.set_mask_parameters()
        mask2d = self.mesh_renderer.render_view(asnumpy=True, asRGB=True)
        nevi_square_mask = self.mesh_renderer.lesion_mask(
            mask2d[:, :, 0], lesion_mask_id=None
        )
        # Set the illumination back to the earlier settings.
        self.set_image_parameters()
        return nevi_square_mask

    def render_body_mask(self):
        body_mask = self.mesh_renderer.body_mask()
        return body_mask

    def set_mask_parameters(self):
        self.mesh_renderer.precompute_light_parameters(
            ambient_color=[1, 1, 1],
            specular_color=[0, 0, 0],
            diffuse_color=[0, 0, 0],
            light_location=self.camera_pos,
        )

        self.mesh_renderer.precompute_material_parameters(
            ambient_color=[1, 1, 1],
            specular_color=[0, 0, 0],
            diffuse_color=[0, 0, 0],
            shininess=0,
        )
        self.mesh_renderer.initialize_renderer()

    def set_image_parameters(self):
        self.mesh_renderer.precompute_light_parameters(
            ambient_color=self.ambient,
            specular_color=self.specular,
            diffuse_color=self.diffuse,
            light_location=self.light_pos,
        )

        self.mesh_renderer.precompute_material_parameters(
            self.ambient,
            self.mat_specular,
            self.mat_diffuse,
            self.shininess,
        )
        self.mesh_renderer.initialize_renderer()

    def render_skin_mask(self):
        self.mesh_renderer.set_texture_image(self.nonskin_texture_mask_tensor)
        self.set_mask_parameters()

        nonskin_mask = self.mesh_renderer.render_view(asnumpy=True, asRGB=True)
        skin_mask = self.mesh_renderer.skin_mask(nonskin_mask[:, :, 0] > 0.5)

        # Set the illumination back to the earlier settings.
        self.set_image_parameters()

        return skin_mask


class Generate2DViews:
    """
    Wrapper for rendering the mesh with blended lesions
    from various view points.
    """

    def __init__(
        self,
        config,
        device,
    ):
        self.bodytex_dir = config["blending"]["bodytex_dir"]
        self.fitz_dir = config["blending"]["fitz_dir"]
        self.annot_dir = config["blending"]["annot_dir"]
        self.num_img = config["generate"]["num_views"]
        self.paste = config["generate"]["paste"]
        self.save_dir = config["generate"]["save_dir"]
        self.mesh_name = config["generate"]["mesh_name"]
        self.tex_dir = config["blending"]["tex_dir"]
        self.anatomy_dir = config["generate"]["anatomy_dir"]
        self.ext = config["blending"]["ext"]
        self.view_size = eval(config["generate"]["view_size"])
        self.background_dir = config["generate"]["background_dir"]
        self.percent_skin = config["generate"]["percent_skin"]
        self.device = device
        self.config = config
        self.skin3d = config["generate"]["skin3d"]
        self.skin3d_annot = config["generate"]["skin3d_annot"]

        self.bodytex_df = pd.read_csv(
            self.skin3d, converters={"scan_id": lambda x: str(x)}
        )
        self.bodytex_ds = BodyTexDataset(
            df=self.bodytex_df,
            dir_textures=self.bodytex_dir,
            dir_annotate=self.skin3d_annot,
        )

        self.background_ds = Background2d(dir_images=self.background_dir)
        self.mesh_filename = os.path.join(
            self.bodytex_dir, self.mesh_name, "model_highres_0_normalized.obj"
        )

        self.fitz_ds = Fitz17KAnnotations(
            dir_images=self.fitz_dir,
            dir_targets=self.annot_dir,
            target_extension=config["extension"]["target_extension"],
            image_extension=config["extension"]["image_extension"],
            color_constancy=shade_of_gray_cc,
        )

        os.makedirs(self.save_dir, exist_ok=True)
        dir_synth_data = os.path.join(
            self.save_dir, "gen_" + self.mesh_name.split("-")[0]
        )
        self.synth_ds = SynthesizeDataset(dir_synth_data)

        self.gen2d = Generate2DHelper(
            mesh_filename=self.mesh_filename,
            dir_blended_textures=self.tex_dir,
            dir_anatomy=self.anatomy_dir,
            fitz_ds=self.fitz_ds,
            background_ds=self.background_ds,
            device=self.device,
            config=self.config,
            debug=False,
            bodytex=self.bodytex_ds,
            blended_file_ext=self.ext,
            percent_skin=self.percent_skin,
        )

    def synthesize_views(self):
        new_params = []
        count_skip = 0
        img_count = 0  # Counts the number of images saved to disk
        pbar = tqdm(total=self.num_img+1, desc="Rendering 2D views")
        while img_count <= self.num_img:
            # keep rendering until num_img are rendered
            success = self.gen2d.randomize_parameters(config=self.config)
            if not success:
                # Checks if the camera/lighting placement works for the random params.
                print("***Camera and lighting placement not successful. Skipping")
                continue
            # Option to paste the lesion.
            paste_img, target = self.gen2d.render_image_and_target(
                paste_lesion=self.paste
            )
            if paste_img is None:
                # Checks if enough skin is visible.
                print("***Not enough skin or unable to paste lesion. Skipping.")
                continue
            target_name = self.synth_ds.generate_target_name()

            # Save image and masks to disk.
            self.synth_ds.save_image(target_name, (paste_img * 255).astype(np.uint8))
            self.synth_ds.save_target(target_name, target)

            # not saving depth for now
            # uncomment if necessary
            # depth_view = target[:,:,4]
            # depth_view = np.clip(depth_view, 0, depth_view.max())
            # depth_view = (depth_view*255/ np.max(depth_view))
            # self.synth_ds.save_depth(target_name, (depth_view).astype(np.uint8))

            # Keep track of the parameters used to generate the image.
            params = {
                "file_id": target_name,
            }
            params.update(self.gen2d.get_params())
            self.synth_ds.update_params(params)
            img_count += 1  # Increament counter.
            pbar.update(1)

        # Save the params to disk.
        self.synth_ds.save_params()
