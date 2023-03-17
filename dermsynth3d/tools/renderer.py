from __future__ import annotations

import math
import numpy as np
import random
import torch

from typing import Optional
from sklearn.preprocessing import normalize

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
)

from dermsynth3d.utils.utils import pix_face_in_set, random_offset
from dermsynth3d.utils.tensor import window_overlap_mask, max_value_in_window


class MeshRendererPyTorch3D:
    """
    Base class for renderer.
    """

    def __init__(
        self,
        mesh,
        device,
        config=None,
        nonskin_face_indexes: Optional[set] = None,
    ):
        self.mesh = mesh
        self.device = device
        self.nonskin_face_indexes = nonskin_face_indexes
        self.up = (1, 0, 0)
        self.fov = 30.0
        self.faces_per_pixel = 1
        self.config = config

        # Set to True to print debug info.
        self.DEBUG = False

        # Compute the 3D coordinates for the center of the face.
        face_vertices = self.mesh.verts_packed()[self.mesh.faces_packed()]
        self.center_face_vertices = face_vertices.mean(axis=1)

        self.dist = None
        self.elev = None
        self.azim = None
        self.at = None
        self.view_size = None
        self.camera_pos = None

        self.cameras = None
        self.raster_settings = None

        self.lights = None

        self.materials = None

        self.renderer = None

        self.rasterizer = None
        self.fragments = None

    def precompute_view_parameters(
        self,
        view_size: tuple[int, int],
        at: tuple[float, float, float],
        dist,
        elev,
        azim,
        camera_pos: Optional[tuple[float, float, float]] = None,
        znear: float = 0.01,
        perspective_correct=False,
        up=None,
    ):
        self.view_size = view_size
        self.at = at

        self.dist = dist
        self.elev = elev
        self.azim = azim
        self.camera_pos = camera_pos
        if self.camera_pos is None:
            eye = None
        else:
            eye = np.asarray(
                [
                    np.asarray(self.camera_pos),
                ]
            )

        if up is None:
            up = np.asarray(
                [
                    np.asarray(self.up),
                ]
            )

        R, T = look_at_view_transform(
            up=up,
            at=np.asarray(
                [
                    np.asarray(self.at),
                ]
            ),
            eye=eye,
            dist=dist,
            elev=elev,
            azim=azim,
        )
        self.R = R
        self.T = T

        if self.DEBUG:
            print(self.at)
            print(self.camera_pos)
            print(self.up)
            print(R)
            print(T)

        self.cameras = FoVPerspectiveCameras(
            device=self.device, R=R, T=T, znear=znear, fov=self.fov
        )

        self.raster_settings = RasterizationSettings(
            image_size=self.view_size,
            blur_radius=0.0,
            faces_per_pixel=self.faces_per_pixel,
            perspective_correct=perspective_correct,
        )

    def precompute_light_parameters(
        self,
        ambient_color=(0.5, 0.5, 0.5),
        specular_color: tuple[float, float, float] = (0.025, 0.025, 0.025),
        diffuse_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
        light_location: Optional[tuple[float, float, float]] = None,
    ):
        if light_location is None:
            if self.camera_pos is None:
                raise ValueError(
                    "Error: Call `self.precompute_view_parameters(...)` first."
                )
            light_location = self.camera_pos

        self.light_location = np.asarray(light_location)
        self.ambient_color = np.asarray(ambient_color)
        self.specular_color = np.asarray(specular_color)
        self.diffuse_color = np.asarray(diffuse_color)

        self.lights = PointLights(
            device=self.device,
            location=np.asarray([self.light_location]),
            ambient_color=np.asarray(
                [
                    self.ambient_color,
                ]
            ),
            diffuse_color=np.asarray(
                [
                    self.diffuse_color,
                ]
            ),
            specular_color=np.asarray(
                [
                    self.specular_color,
                ]
            ),
        )

    def precompute_material_parameters(
        self,
        ambient_color=(0.5, 0.5, 0.5),
        specular_color: tuple[float, float, float] = (0.025, 0.025, 0.025),
        diffuse_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
        shininess: float = 50.0,
    ):
        self.shininess = shininess
        self.ambient_color = np.asarray(ambient_color)
        self.specular_color = np.asarray(specular_color)
        self.diffuse_color = np.asarray(diffuse_color)

        self.materials = Materials(
            device=self.device,
            specular_color=np.asarray(
                [
                    self.specular_color,
                ]
            ),
            shininess=self.shininess,
        )

    def camera_parameters_offset_face(self, face_idx, normal_weight=0.1):
        # A small value to offset the camera_pos.
        # If the face_normal contains 0's (e.g., [1, 0, 0])
        # then get an error when computing the rotation matrix.
        # To prevent this, add a small offset to the camera position.
        eps = 0.0001
        face_coords = self.center_face_vertices[face_idx, :]
        face_coords = face_coords.cpu().detach().numpy()

        face_normal = self.mesh.faces_normals_packed()[face_idx, :]
        face_normal = face_normal.cpu().detach().numpy()
        # Occasional faces do not seem to be normalized.
        face_normal = normalize([face_normal])[0]
        camera_pos = face_coords + (face_normal * normal_weight) + eps
        at = face_coords

        return camera_pos, at

    def randomize_view_parameters(
        self,
        dists=(0.3, 3),
        elevs=(90, 270),
        azims=(0, 270),
        ats=(0.25, 1.75),
        ambients=(0.2, 0.99),
        speculars=(0, 0.1),
        diffuses=(0.2, 0.99),
        shininess=(30.0, 60.0),
        view_size=(512, 512),  # View size is not randomized.
        surface_offset_weight=(0.1, 0.2),
        face_idx=None,
        camera_pos=None,
        look_at=None,
        znear=1,
    ):
        normal_weight = None
        dist = None
        elev = None
        azim = None

        if look_at is not None:
            at = look_at
        elif surface_offset_weight is not None:
            # Choose a random face.
            if face_idx is None:
                face_idx = np.random.randint(0, self.center_face_vertices.shape[0])

            b = surface_offset_weight[1]
            a = surface_offset_weight[0]
            normal_weight = (b - a) * np.random.random_sample() + a
            camera_pos, at = self.camera_parameters_offset_face(
                face_idx, normal_weight=normal_weight
            )

        else:
            # Randomize views.
            dist = np.round(random.uniform(dists[0], dists[1]), 2)
            elev = np.round(random.uniform(elevs[0], elevs[1]), 2)
            azim = np.round(random.uniform(azims[0], azims[1]), 2)
            at = (np.round(random.uniform(ats[0], ats[1]), 2), 0, 0)
            camera_pos = None

        self.precompute_view_parameters(
            view_size=view_size,
            at=at,
            dist=dist,
            elev=elev,
            azim=azim,
            camera_pos=camera_pos,
            znear=znear,
        )

        # Randomize lighting.
        ambient = [np.round(random.uniform(ambients[0], ambients[1]), 2)] * 3
        specular = [np.round(random.uniform(speculars[0], speculars[1]), 2)] * 3
        diffuse = [np.round(random.uniform(diffuses[0], diffuses[1]), 2)] * 3
        shininess = np.round(random.uniform(shininess[0], shininess[1]), 2)

        light_location = self.camera_pos

        self.precompute_light_parameters(
            ambient, specular, diffuse, light_location=light_location
        )

        self.precompute_material_parameters(ambient, specular, diffuse, shininess)

        self.params = {
            "dist": dist,
            "elev": elev,
            "azim": azim,
            "look_at": list(at),
            "ambient": ambient,
            "specular": specular,
            "diffuse": diffuse,
            "shininess": shininess,
            "camera_pos": list(camera_pos),
            "normal_weight": normal_weight,
            "light_pos": light_location,
        }

        self.initialize_renderer()
        self.compute_fragments()

    def initialize_renderer(self):
        if self.cameras is None:
            raise ValueError(
                "Error: call `self.precompute_view_parameters(...)` first."
            )

        if self.lights is None:
            raise ValueError(
                "Error: call `self.precompute_light_parameters(...)` first."
            )

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=self.lights,
                materials=self.materials,
            ),
        )

    def render_view(self, asnumpy=False, asRGB=False, isclip=True):
        """Returns the rendered 2D view.

        Args:
            asnumpy (bool, optional): If True, returns a detached numpy array.
                If False, returns an attached tensor. Defaults to False.
            asRGB (bool, optional): If True, returns only the RGB component.
                If False, returns the full tensor.
            isclip (bool, optional): If True, clips values between 0 and 1.

        Raises:
            ValueError: Raised if initialization is not yet called.

        Returns:
            array or tensor: Rendered 2D view.
                Datatype depends on `asnumpy` flag.
        """
        if self.renderer is None:
            raise ValueError("Error: call `self.initialize_renderer() first.")

        images = self.renderer(self.mesh)
        if isclip:
            images = torch.clip(images, 0, 1)

        if asnumpy:
            images = images.cpu().detach().numpy()

        if asRGB:
            images = images[:, :, :, :3].squeeze()

        return images

    def compute_fragments(self):
        if self.cameras is None:
            raise ValueError("Error: call `self.precompute_view_parameters(...) first.")

        self.rasterizer = MeshRasterizer(
            cameras=self.cameras, raster_settings=self.raster_settings
        )
        self.fragments = self.rasterizer(self.mesh)

    def pixel_faces_in_set(self, face_indexes):
        self.check_raise_fragments()

        pix2face = self.fragments.pix_to_face.squeeze().cpu().detach().numpy()
        mask = pix_face_in_set(pix2face, face_indexes)

        return mask

    def set_texture_image(self, texture_image):
        """Sets the texture image using the existing UV mapping.

        This can be used render the same view with different textures.

        Args:
            texture_image (torch.float32): H x W x C texture image.
        """
        texturesuv = TexturesUV(
            maps=[texture_image],
            faces_uvs=self.mesh.textures.faces_uvs_padded(),
            verts_uvs=self.mesh.textures.verts_uvs_padded(),
        )
        self.mesh.textures = texturesuv

    def render_lesion_mask(self, asnumpy=False, asRGB=False):
        self.precompute_light_parameters(
            ambient_color=[1, 1, 1],
            specular_color=[0, 0, 0],
            diffuse_color=[0, 0, 0],
            light_location=None,
        )
        self.precompute_material_parameters(
            ambient_color=[1, 1, 1],
            specular_color=[0, 0, 0],
            diffuse_color=[0, 0, 0],
            shininess=10.0,
        )
        self.initialize_renderer()
        images = self.render_view(asnumpy, asRGB)
        return images

    def body_mask(self):
        mask = np.asarray(self.fragments.zbuf.squeeze().cpu() > 0)
        return mask

    def lesion_mask(self, lesion_view2d, lesion_mask_id=None, tol=0.1):
        if lesion_mask_id is None:
            lower = (1 - tol) / 255
            upper = (255) / 255
        else:
            lower = (lesion_mask_id - tol) / 255
            upper = (lesion_mask_id + tol) / 255

        mask = np.logical_and(lesion_view2d >= lower, lesion_view2d <= upper)
        return mask * self.body_mask()

    def skin_mask(self, nonskin_mask):
        return self.body_mask() * ~nonskin_mask

    def pixels_to_face(self, asnumpy=True):
        """Returns the face indexes for each pixel.

        Positive values indicate the indexes of the face.
        Negative values indicate background.

        Args:
            asnumpy (bool, optional): If True, returns a detached numpy array.
                Defaults to True.

        Returns:
            _type_: _description_
        """
        self.check_raise_fragments()
        pix2face = self.fragments.pix_to_face
        if asnumpy:
            pix2face = pix2face.squeeze().cpu().detach().numpy()

        return pix2face

    def face_indexes_of_mask(self, mask, asnumpy=True):
        pix2face = self.pixels_to_face(asnumpy=asnumpy)
        face_indexes_of_mask = pix2face[mask > 0]
        return face_indexes_of_mask

    def uvs_of_mask(self, mask, asnumpy=True):
        """
        Returns the normalized UVs within the binary mask for the view.
        """
        face_indexes_of_mask = self.face_indexes_of_mask(mask, asnumpy=asnumpy)

        # UV coordinates per-vertex
        verts_uvs = self.mesh.textures.verts_uvs_padded().squeeze()

        # Index into verts_uvs for each face.
        faces_uvs = self.mesh.textures.faces_uvs_padded().squeeze()

        # UV coordinates for each face of the lesion.
        uvs_per_lesion_face = verts_uvs[faces_uvs[face_indexes_of_mask]]

        # uv*2-1 converts to pixel space.
        uvs_per_lesion_face = (uvs_per_lesion_face * 2 - 1).unsqueeze(0)
        return uvs_per_lesion_face

    def view_uvs(self, asnumpy=True):
        verts_uvs = self.mesh.textures.verts_uvs_padded().squeeze()

        # Index into verts_uvs for each face.
        faces_uvs = self.mesh.textures.faces_uvs_padded().squeeze()

        uvs_per_face = verts_uvs[faces_uvs]

        pix2faces = self.pixels_to_face(asnumpy=False)
        # If multiple faces per pixels are used, then this part will fail.
        pix2face = pix2faces.squeeze()  # [0,:,:,face_idx]
        uvs_per_face_flat = uvs_per_face[pix2face.view(-1)]

        bary_coords = self.fragments.bary_coords.squeeze()

        uvs_per_face_view = uvs_per_face_flat.reshape(
            (pix2face.shape[0], pix2face.shape[1], 3, 2)
        )

        # Weigh uvs by barycentric weights.
        out = torch.matmul(bary_coords.unsqueeze(2), uvs_per_face_view)

        if asnumpy:
            out = out.squeeze().cpu().numpy()

        return out

    def skin_face_indexes(self):
        """
        Return the indexes of the faces that contain skin.
        """
        if self.nonskin_face_indexes is None:
            raise ValueError("Error: `self.nonskin_face_indexes must be set first.")
        all_face_indexes = set(np.arange(0, self.mesh.faces_packed().shape[0]))
        skin_face_indexes = all_face_indexes - self.nonskin_face_indexes

        return skin_face_indexes

    def skin_overlap_with_window(self, window_size: tuple[int, int]):
        """
        Pixels where the window overlaps with all skin.
        """
        skin_mask = self.skin_mask()
        overlap_with_skin = window_overlap_mask(
            skin_mask, window_size=window_size, pad_value=0, output_type="all_ones"
        )

        return overlap_with_skin

    def depth_view(self, asnumpy=True):
        """
        Returns the depth of the view.

        Args:
            asnumpy (bool, optional): If True, return a detached numpy array.
                Defaults to True.

        Returns:
            array or tensor: Depth from the camera for each pixel.
                A negative value indicates background.
        """
        self.check_raise_fragments()
        depth_view = self.fragments.zbuf
        if asnumpy:
            return depth_view.squeeze().cpu().detach().numpy()
        else:
            return depth_view

    def view_center_face_coordiantes(self):
        """
        Returns the face's center coordinates within the view.
        """
        pix2face = self.pixels_to_face(asnumpy=True)
        pix2face_flat = pix2face[pix2face > 0]
        vertices = self.center_face_vertices[pix2face_flat]
        return vertices

    def max_difference_in_window_from_dist(
        self, target_dist: float, window_size: tuple[int, int]
    ):
        depth_view = self.depth_view()
        max_window_img = max_value_in_window(
            np.abs(depth_view - target_dist),
            window_size,
            pad_value=-1,
        )
        return max_window_img.squeeze()

    def depth_skin_view(self, skin_mask):
        return self.depth_view() * skin_mask + (skin_mask - 1)

    def max_depth_difference_to_target(
        self,
        lesion_mask,
        skin_mask,
        target_value: Optional[float] = None,
    ):
        depth_skin_view = self.depth_skin_view(skin_mask)
        if target_value is None:
            target_value = self.params["normal_weight"]

        return (np.abs(depth_skin_view - target_value) * lesion_mask).max()

    def check_raise_fragments(self):
        """
        Raises an error if fragments are not computed.
        """
        if self.fragments is None:
            raise ValueError("Error: Call `self.compute_fragments(...) first.")

    def faces_is_skin(self, nonskin_texture_mask_tensor):
        uvs_of_faces = self.mesh.textures.verts_uvs_padded().squeeze()[
            self.mesh.textures.faces_uvs_padded().squeeze()
        ]
        texture_coords_of_faces = uvs_of_faces * nonskin_texture_mask_tensor.shape[0]
        texture_coords_of_faces = (
            texture_coords_of_faces.cpu().numpy().round().astype(np.int32)
        )

        img_size = nonskin_texture_mask_tensor.shape[:2]
        mask_texture_recon = np.zeros(shape=img_size)
        nonskin_mask = nonskin_texture_mask_tensor.cpu().detach().numpy()
        is_face_skin = np.ones(shape=len(texture_coords_of_faces), dtype=np.int32)
        for idx, coords in enumerate(texture_coords_of_faces):
            skin_coord = 1
            for coord in coords:
                if nonskin_mask[img_size[0] - coord[1], coord[0], 0] == 1:
                    mask_texture_recon[img_size[0] - coord[1], coord[0]] = 1
                    skin_coord = 0
                    break

            is_face_skin[idx] = skin_coord

        return is_face_skin

    def anatomy_image(self, anatomy_vert_labels):
        """
        Returns the anatomy labels in the image.
        """
        pix2face = self.pixels_to_face()
        pix2vert = self.mesh.faces_padded().squeeze()[pix2face]
        pix2vert = pix2vert.detach().cpu().numpy()
        # This uses the single vertex of a face to assign a label.
        # Not ideal. Could be changed to the most frequent label?
        anatomy_image = anatomy_vert_labels[pix2vert[:, :, 0]]
        anatomy_image = anatomy_image * self.body_mask()
        return anatomy_image


def camera_world_position(dist, elev, azim, at, degrees=True):
    """
    Returns a rotation and translational matrix in world view.
    """
    if degrees:
        elev = math.pi / 180.0 * elev
        azim = math.pi / 180.0 * azim
    x = dist * np.cos(elev) * np.sin(azim)
    y = dist * np.sin(elev)
    z = dist * np.cos(elev) * np.cos(azim)

    return np.asarray([x, y, z]) + at


def camera_pos_from_normal(look_at, normal, normal_weight, eps=0.0001):
    # Ensure normal.
    normal = normalize([normal])[0]

    # Randomly choose a weight to apply to the normal.
    # normal_weight = random.uniform(normal_weight_range[0], normal_weight_range[1])

    # Camera position is a weighted offset from the `look_at` position.
    # We add a small value `eps` as otherwise can get errors computing
    # the rotation matrix when a normal contains a 0 e.g. [1, 0, 0].
    camera_pos = look_at + (normal * normal_weight) + eps

    return camera_pos
