from scipy import ndimage
from PIL import Image

import numpy as np

import torch
import torchvision
import pytorch3d


from dermsynth3d.deepblend.utils import (
    single_channel_to_rgb_tensor,
    numpy2tensor,
    laplacian_filter_tensor,
)
from dermsynth3d.utils.utils import (
    mask2boundingbox,
    random_offset,
    random_bound,
)
from dermsynth3d.models.model import Vgg16
from dermsynth3d.losses.deepblend_loss import (
    total_variation_loss,
    gradient_loss,
    style_gram_loss,
)
from dermsynth3d.utils.textures import UVViewMapper
from dermsynth3d.utils.image import simple_augment, float_img_to_uint8, uint8_to_float32
from dermsynth3d.tools.renderer import (
    camera_pos_from_normal,
)


def texture_mask_of_lesion_mask_id(texture_mask, lesion_mask_id: int, device):
    mask_lesion = texture_mask.cpu().detach().numpy() == lesion_mask_id / 255
    lesion_id_texture_mask = torch.tensor(
        mask_lesion, dtype=torch.float32, device=device
    )
    return lesion_id_texture_mask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PasteTextureImage:
    """
    Pastes lesions into texture images and create masks.
    """

    def __init__(
        self,
        original_texture_image_tensor,
        nonskin_texture_mask_tensor,
        mesh_renderer,
        view_size,
        n_samples: int = 10000,
    ):
        self.original_texture_image_tensor = original_texture_image_tensor
        self.nonskin_texture_mask_tensor = nonskin_texture_mask_tensor
        self.mesh_renderer = mesh_renderer
        self.view_size = view_size
        self.blend = None
        self.blend_dilated = None

        self.max_depth_diff = None
        self.mask_id = None

        # Sample points on the skin.
        self._sample_skin_points(n_samples)
        # Keeps track of the index of the random samples.
        self.sample_counter = -1

        # Initialize textures and masks that we'll use to paste the lesions.
        self.initialize_pasted_masks_and_textures()

    def view_params(self):
        p = {
            "look_at": list(self.sample_look_at),
            "normal": list(self.sample_normal),
            "camera_pos": list(self.sample_camera_pos),
            "normal_weight": self.normal_weight,
            "max_depth_diff": self.max_depth_diff,
            "lesion_mask_id": self.mask_id,
        }
        return p

    def sample_next_view(self, surface_offset_bounds=(0.4, 0.6)):
        """Sample a random view with the center pixel containing skin."""
        self.sample_counter += 1
        # If out-of-bounds, may need to increase `n_samples`.
        # Or there may not be a suitable location to paste.
        sample_idx = self.sample_skin_indexes[self.sample_counter]

        self.sample_look_at = self.sample_coords[sample_idx]
        self.sample_normal = self.sample_normals[sample_idx]

        self.normal_weight = random_bound(*surface_offset_bounds)

        # Determine the camera position.
        self.sample_camera_pos = camera_pos_from_normal(
            look_at=self.sample_look_at,
            normal=self.sample_normal,
            normal_weight=self.normal_weight,
        )

        self.set_renderer_view(
            face_idx=None,
            surface_offset_range=None,
            look_at=self.sample_look_at,
            camera_pos=self.sample_camera_pos,
        )

    def _sample_skin_points(self, n_samples: int = 10000):
        """Samples points on the mesh and determines if the point is on skin."""

        # Set texture to the non-skin mask.
        self.mesh_renderer.set_texture_image(self.nonskin_texture_mask_tensor)

        coords, normals, textures = pytorch3d.ops.sample_points_from_meshes(
            meshes=self.mesh_renderer.mesh,
            num_samples=n_samples,
            return_normals=True,
            return_textures=True,
        )

        sample_textures = textures.cpu().detach().numpy().squeeze()
        # `sample_textures` will have [1,1,1] for each pixel that is non-skin.
        # So if the sum < 3, then is a skin pixel.
        samples_is_skin = sample_textures.sum(axis=1) < 3
        # Randomly permute the skin indexes - likely not necessary as these are already sampled.
        self.sample_skin_indexes = np.random.permutation(np.where(samples_is_skin)[0])

        # Spatial coordinates of the surface of the mesh.
        self.sample_coords = coords.cpu().detach().numpy().squeeze()
        # Normals corresponding to the surface points.
        self.sample_normals = normals.cpu().detach().numpy().squeeze()

    def paste_on_texture(self, img, mask, mask_id: int, depth_diff_thresh=0.02):
        self.max_depth_diff = None
        self.mask_id = None

        self.init_target_lesion(img, mask)
        max_depth_diff = self.lesion_max_depth_diff()
        if max_depth_diff > depth_diff_thresh:
            return "Skipping sample {} due to high depth change".format(
                self.sample_counter
            )

        accepted = self.paste_masks_and_texture_images(mask_id)
        if not accepted:
            return "Skipping sample {} as overlaps with existing lesion.".format(
                self.sample_counter
            )

        self.max_depth_diff = max_depth_diff
        self.mask_id = mask_id
        return None

    def original_view(self):
        self.mesh_renderer.set_texture_image(self.original_texture_image_tensor)
        view2d = self.mesh_renderer.render_view(asnumpy=True, asRGB=True)
        return view2d

    def nonskin_mask(self):
        self.mesh_renderer.set_texture_image(self.nonskin_texture_mask_tensor)
        nonskin_mask = self.mesh_renderer.render_view(asnumpy=True, asRGB=True)
        return nonskin_mask

    def skin_mask(self):
        nonskin_mask = self.nonskin_mask()
        skin_mask = self.mesh_renderer.skin_mask(nonskin_mask[:, :, 0] > 0.5)
        return skin_mask

    def lesion_max_depth_diff(self):
        view_seg_dilated = self.blend_dilated.view_seg()
        skin_mask = self.skin_mask()
        target_value = None

        max_depth_diff = self.mesh_renderer.max_depth_difference_to_target(
            view_seg_dilated, skin_mask, target_value=target_value
        )
        return max_depth_diff

    def set_renderer_view(
        self,
        face_idx,
        surface_offset_range=(0.4, 0.6),
        look_at=None,
        camera_pos=None,
    ):
        self.mesh_renderer.randomize_view_parameters(
            ambients=(1, 1),
            speculars=(0, 0),
            diffuses=(0, 0),
            surface_offset_weight=surface_offset_range,
            face_idx=face_idx,
            view_size=self.view_size,
            znear=1.0,  # SET TO 1 for DEEP BLENDING. 0.01 gives artefacts.
            look_at=look_at,
            camera_pos=camera_pos,
        )

    def init_target_lesion(
        self,
        lesion_img,
        lesion_seg,
        augment=True,
        resize=True,
    ):
        """Set the 2D lesion and mask."""

        if augment:
            lesion_img, lesion_seg = simple_augment(lesion_img, lesion_seg)

        if resize:
            # Resize as some lesions are very large
            # relative to the view size we are taking.
            # Only allow the lesion to be a third of the size of the view.
            new_size = np.asarray(self.view_size) // 3
            lesion_img = Image.fromarray(float_img_to_uint8(lesion_img))
            lesion_seg = Image.fromarray(float_img_to_uint8(lesion_seg))
            lesion_img.thumbnail(size=new_size, resample=Image.BILINEAR)
            lesion_seg.thumbnail(size=new_size, resample=Image.NEAREST)
            lesion_img = uint8_to_float32(np.asarray(lesion_img))
            lesion_seg = uint8_to_float32(np.asarray(lesion_seg))
            # We need even dimensions.
            # Force this by dropping the last row/col if not even.
            if lesion_seg.shape[0] % 2:
                lesion_img = lesion_img[:-1, :, :]
                lesion_seg = lesion_seg[:-1, :]

            if lesion_seg.shape[1] % 2:
                lesion_img = lesion_img[:, :-1, :]
                lesion_seg = lesion_seg[
                    :,
                    :-1,
                ]

        self.blend = Blend(
            lesion_img=lesion_img,
            lesion_seg=lesion_seg,
            view_size=self.view_size,
        )
        lesion_seg_dilated = ndimage.binary_dilation(
            self.blend.lesion_seg, iterations=8
        )
        self.blend_dilated = Blend(
            lesion_img=lesion_img,
            lesion_seg=lesion_seg_dilated,
            view_size=self.view_size,
        )

    def pasted_image(self):
        view2d = self.original_view()
        return self.blend.view_lesion(view2d)

    def initialize_pasted_masks_and_textures(self):
        self.original_texture_np = np.asarray(
            self.original_texture_image_tensor.cpu().detach() * 255, dtype=np.uint8
        )
        self.pasted_texture = self.original_texture_np.copy()
        self.pasted_dilated_texture = self.original_texture_np.copy()

        self.lesion_mask = np.zeros(
            shape=(
                self.original_texture_np.shape[0],
                self.original_texture_np.shape[1],
                1,
            ),
            dtype=np.uint8,
        )
        self.lesion_dilated_mask = self.lesion_mask.copy()

    def paste_masks_and_texture_images(self, lesion_mask_id):
        view2d = self.original_view()
        uv_view_mapper_dilated = UVViewMapper(
            view_uvs=self.mesh_renderer.view_uvs(),
            paste_img=self.blend_dilated.view_lesion(view2d),
            body_mask_view=self.mesh_renderer.body_mask(),
            lesion_mask_view=self.blend_dilated.view_seg(),
            texture_img_size=4096,
        )
        mask_pad4_dilated, tex_pad4_dilated = uv_view_mapper_dilated.texture_pad(
            seam_thresh=0.1, niter=4
        )

        partial_pad_pasted_texture = np.asarray(tex_pad4_dilated * 255, dtype=np.uint8)

        partial_lesion_pad_mask_dilated = (
            uv_view_mapper_dilated.padder.padded_lesion_mask(mask_pad4_dilated)
        )

        if sum(self.lesion_dilated_mask[partial_lesion_pad_mask_dilated]) > 0:
            # Overlaps with existing lesion. Do not proceed.
            return False

        self.pasted_dilated_texture = (
            partial_pad_pasted_texture * partial_lesion_pad_mask_dilated
        ) + (self.pasted_dilated_texture * ~partial_lesion_pad_mask_dilated)

        self.lesion_dilated_mask[partial_lesion_pad_mask_dilated] = lesion_mask_id

        uv_view_mapper = UVViewMapper(
            view_uvs=self.mesh_renderer.view_uvs(),
            paste_img=self.blend.view_lesion(view2d),
            body_mask_view=self.mesh_renderer.body_mask(),
            lesion_mask_view=self.blend.view_seg(),
            texture_img_size=4096,
        )

        mask_pad4, tex_pad4 = uv_view_mapper.texture_pad(seam_thresh=0.1, niter=4)
        partial_pasted_texture = np.asarray(tex_pad4 * 255, dtype=np.uint8)
        partial_pasted_lesion_mask = uv_view_mapper.padder.padded_lesion_mask(mask_pad4)
        self.pasted_texture = (partial_pasted_texture * partial_pasted_lesion_mask) + (
            self.pasted_texture * ~partial_pasted_lesion_mask
        )
        self.lesion_mask[partial_pasted_lesion_mask] = lesion_mask_id
        return True


class DeepTextureBlend3d:
    def __init__(
        self,
        blended3d,
        mesh_renderer,
        deepblend,
        device,
        view_size=None,
    ):
        self.blended3d = blended3d
        self.mesh_renderer = mesh_renderer
        self.deepblend = deepblend
        self.device = device
        self.view_size = view_size
        if self.view_size is None:
            self.view_size = (512, 512)

        # Original texture image.
        self.original_texture = self.blended3d.texture_image(astensor=True).to(self.device)
        # Mask for the texture image.
        self.texture_mask = self.blended3d.lesion_texture_mask(astensor=True).to(self.device)
        # Pasted lesion on the texture image.
        self.pasted_texture = self.blended3d.pasted_texture_image(astensor=True).to(self.device)
        # Lesion with expanded borders on the texture image.
        self.dilated_texture = self.blended3d.dilated_texture_image(astensor=True).to(self.device)

        # This is the texture image we are blending on.
        # Load once so can blend multiple lesions.
        self.texture_image = self.pasted_texture.clone().detach().contiguous().to(self.device)
        self.texture_image.requires_grad = True

    def set_params(self, params):
        self.params = params
        self.lesion_id_texture_mask = texture_mask_of_lesion_mask_id(
            self.texture_mask, params.lesion_mask_id, self.device
        )

    def randomize_view_offset(self, offset=None):
        # View with a random distance offset.
        if offset is None:
            offset = random_offset()

        normal_weight = self.params.normal_weight
        lower = normal_weight - offset
        upper = normal_weight + offset
        surface_offset_weight = [lower, upper]

        look_at = None
        camera_pos = None
        face_idx = None
        if "face_idx" in self.params:
            face_idx = int(self.params.face_idx)

        if "look_at" in self.params:
            # If 'look_at' exists in the params,
            # then override the `face_idx` option
            # and instead use `look_at` with `camera_pos`
            # to determine the view to render.
            face_idx = None
            surface_offset_weight = None
            look_at = np.asarray(self.params["look_at"])
            normal = np.asarray(self.params["normal"])
            camera_pos = camera_pos_from_normal(
                look_at=look_at,
                normal=normal,
                normal_weight=normal_weight,
            )

        self.mesh_renderer.randomize_view_parameters(
            ambients=(1, 1),
            speculars=(0, 0),
            diffuses=(0, 0),
            surface_offset_weight=surface_offset_weight,
            face_idx=face_idx,
            view_size=self.view_size,
            znear=1.0,
            look_at=look_at,
            camera_pos=camera_pos,
        )

    def render_texture_views(self, pad: int = 32):
        tensors, gt_gradient = render_views_with_textures(
            self.mesh_renderer,
            self.texture_image,
            self.lesion_id_texture_mask,
            self.pasted_texture,
            self.dilated_texture,
            self.original_texture,
            lesion_mask_id=None,  # params.lesion_mask_id,
            pad=pad,
        )

        return tensors, gt_gradient

    def compute_loss_of_random_offset_view(self, pad: int = 32):
        # Blended image. We are learning this. Clamp to acceptable range.
        self.texture_image.data.clamp_(0, 1)
        # Randomize the offset of the view.
        self.randomize_view_offset(offset=None)

        tensors, gt_gradient = self.render_texture_views(pad)
        loss = self.deepblend.loss(
            tensors["composite"],
            tensors["original"],
            tensors["mask"],
            tensors["pasted"],
            gt_gradient,
        )

        return loss

    def postprocess_blended_texture_image(self):
        """
        After we blend the texture image, artefacts can still occur,
        especially at the border of the blended texture.

        As well, the seams are not padded during the blending process.

        This function combines the blended texture image with
        the original texture image and performs seam padding as needed.

        In this part, we:
        - replace the textures outside of the lesion mask
        with the original textures to replace border artefacts
        - perform texture padding for the blended lesion
        (only needed if blending across seams)

        Returns:
            The processed texture image with blended lesions.
        """

        # Initialize the merged texture as the original texture.
        merged_texture_np = np.asarray(self.blended3d.texture_image())

        # Clamp to acceptable range.
        self.texture_image.data.clamp_(0, 1)
        blended_texture_np = (self.texture_image.detach().cpu().numpy() * 255).astype(
            np.uint8
        )

        # Mask of the lesions.
        texture_mask_np = (
            np.asarray(self.texture_mask.cpu().detach()[:, :, np.newaxis]) * 255
        )

        params_df = self.blended3d.lesion_params()

        for row_idx, params in params_df.iterrows():
            face_idx = None
            surface_offset_weight = None

            normal_weight = params.normal_weight

            if "face_idx" in params:
                face_idx = int(params["face_idx"])
                surface_offset_weight = [normal_weight, normal_weight]

            if "look_at" in params:
                # Override face_idx if look_at is given.
                face_idx = None
                surface_offset_weight = None
                look_at = np.asarray(params["look_at"])
                normal = np.asarray(params["normal"])
                camera_pos = camera_pos_from_normal(
                    look_at=look_at,
                    normal=normal,
                    normal_weight=normal_weight,
                )

            lesion_mask_id = params.lesion_mask_id
            # No offset in the surface_weight.
            self.mesh_renderer.randomize_view_parameters(
                ambients=(1, 1),
                speculars=(0, 0),
                diffuses=(0, 0),
                surface_offset_weight=surface_offset_weight,
                face_idx=face_idx,
                view_size=self.view_size,
                znear=1.0,
            )

            # Set to the blended texture image.
            self.mesh_renderer.set_texture_image(texture_image=self.texture_image)

            # Blended image.
            img = self.mesh_renderer.render_view(asnumpy=True, asRGB=True)

            # Texture mask for only the `lesion_mask_id`
            lesion_id_texture_mask = texture_mask_of_lesion_mask_id(
                self.texture_mask, lesion_mask_id, self.device
            )

            # Render the texture mask for the specific lesion.
            self.mesh_renderer.set_texture_image(
                texture_image=lesion_id_texture_mask[:, :, np.newaxis]
            )
            mask2d = self.mesh_renderer.render_lesion_mask(asnumpy=True, asRGB=True)

            # Remove background on mask.
            lesion_mask = mask2d * self.mesh_renderer.body_mask()[:, :, np.newaxis]
            lesion_mask = (lesion_mask[:, :, 0] > 0.5) * 1

            # Pad if across seams.
            uv_view_mapper = UVViewMapper(
                view_uvs=self.mesh_renderer.view_uvs(),
                paste_img=img,  # Blended image.
                body_mask_view=self.mesh_renderer.body_mask(),
                lesion_mask_view=lesion_mask,
                texture_img_size=4096,
            )
            mask_pad4, tex_pad4 = uv_view_mapper.texture_pad(seam_thresh=0.1, niter=4)
            lesion_pad_channel = uv_view_mapper.padder.LESION_PAD_CHANNEL
            padded_blended_texture = np.asarray(tex_pad4 * 255, dtype=np.uint8)
            lesion_pad_mask = mask_pad4[:, :, lesion_pad_channel] == 1
            lesion_pad_mask = lesion_pad_mask[:, :, np.newaxis]
            texture_mask_lesion = texture_mask_np == lesion_mask_id

            # Use the blended lesions for the lesion mask,
            # use the earlier iteration of the blended_texture for non-masked.
            merged_texture_np = (blended_texture_np * texture_mask_lesion) + (
                merged_texture_np * ~texture_mask_lesion
            )

            # Call this after to overwrite the lesion padding at the seams.
            # Use padded blended textures for lesion padded areas,
            merged_texture_np = (padded_blended_texture * lesion_pad_mask) + (
                merged_texture_np * ~lesion_pad_mask
            )

        return merged_texture_np


class DeepImageBlend:
    def __init__(
        self,
        normalize=None,
        gpu_id=0,
        grad_weight=100000,
        style_weight=1000000,
        content_weight=2,
        tv_weight=0.0001,
    ):
        self.normalize = normalize
        self.gpu_id = gpu_id

        if self.normalize is None:
            self.normalize = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

        self.model = Vgg16().to(gpu_id)

        self.mse = torch.nn.MSELoss()

        # Loss function weights.
        # These are set empirically such that each term
        # roughly contributes the same amount to the loss.
        self.grad_weight = grad_weight
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight

        self.losses = []

    def loss(
        self,
        composite_img_tensor,
        original_img_tensor,
        lesion_mask_tensor,
        pasted_img_tensor,
        gt_gradient,
    ):
        grad_loss = self.gradient_loss(composite_img_tensor, gt_gradient)
        content_loss = self.content_loss(
            composite_img_tensor, lesion_mask_tensor, pasted_img_tensor
        )
        style_loss = self.style_loss(composite_img_tensor, original_img_tensor)
        tv_loss = self.total_variation_loss(composite_img_tensor)
        total_loss = grad_loss + content_loss + style_loss + tv_loss

        self.losses.append(
            {
                "grad": grad_loss.item(),
                "content": content_loss.item(),
                "style": style_loss.item(),
                "tv": tv_loss.item(),
                "loss": total_loss.item(),
            }
        )
        return total_loss

    def gradient_loss(
        self,
        composite_img_tensor,
        gt_gradient,
    ):
        grad_loss = gradient_loss(
            composite_img_tensor, gt_gradient, self.mse, self.gpu_id
        )
        grad_loss *= self.grad_weight
        return grad_loss

    def content_loss(
        self,
        composite_img_tensor,
        lesion_mask_tensor,
        pasted_img_tensor,
    ):
        blend_object_features = self.model(
            self.normalize(composite_img_tensor * lesion_mask_tensor)
        )
        source_object_features = self.model(
            self.normalize(pasted_img_tensor * lesion_mask_tensor)
        )
        # Assumes VGG16 here.
        content_loss = self.mse(
            blend_object_features.relu2_2, source_object_features.relu2_2
        )
        content_loss *= self.content_weight
        return content_loss

    def style_loss(
        self,
        composite_img_tensor,
        original_img_tensor,
    ):
        blend_features_style = self.model(self.normalize(composite_img_tensor))
        original_features_style = self.model(self.normalize(original_img_tensor))

        # Assumes VGG features.
        style_loss = style_gram_loss(
            original_features_style, blend_features_style, self.mse
        )
        style_loss *= self.style_weight
        return style_loss

    def total_variation_loss(self, composite_img_tensor):
        tv_loss = total_variation_loss(composite_img_tensor)
        tv_loss *= self.tv_weight
        return tv_loss


class Blend:
    def __init__(
        self,
        lesion_img,
        lesion_seg,
        view_size=(512, 512),
        x_start=None,
        y_start=None,
    ):
        self._lesion_img = lesion_img
        self.lesion_seg = lesion_seg
        self.view_size = view_size
        self.x_start = x_start
        self.y_start = y_start

        if self.x_start is None:
            self.x_start = self.view_size[0] // 2

        if self.y_start is None:
            self.y_start = self.view_size[1] // 2

    def lesion_seg_dilated(self, iterations=3):
        return ndimage.binary_dilation(self.lesion_seg, iterations=iterations)

    def view_seg_dilated(self, iterations=3, asrgbtensor=False):
        lesion_seg_dilated = self.lesion_seg_dilated(iterations)
        view_seg_dilated = paste_blend(
            x_start=self.x_start,
            y_start=self.y_start,
            img=np.ones(shape=self.lesion_seg.shape, dtype=np.float32),
            mask=lesion_seg_dilated > 0,
            canvas=np.zeros(shape=self.view_size, dtype=np.float32),
        )

        if asrgbtensor:
            view_seg_dilated = single_channel_to_rgb_tensor(view_seg_dilated)

        return view_seg_dilated

    def view_seg(self, asrgbtensor=False):
        view_seg = paste_blend(
            x_start=self.x_start,
            y_start=self.y_start,
            img=np.ones(shape=self.lesion_seg.shape, dtype=np.float32),
            mask=self.lesion_seg > 0,
            canvas=np.zeros(shape=self.view_size, dtype=np.float32),
        )

        if asrgbtensor:
            view_seg = single_channel_to_rgb_tensor(view_seg)

        return view_seg

    def view_lesion(self, view2d):
        return paste_blend(
            x_start=self.x_start,
            y_start=self.y_start,
            img=self._lesion_img,
            mask=self.lesion_seg[:, :, np.newaxis] > 0,
            canvas=view2d,
        )

    def lesion_img(self, astensor=False):
        if astensor:
            return numpy2tensor(self._lesion_img, gpu_id=device)

        return self._lesion_img

    def lesion_mask(self, asrgbtensor=False):
        if asrgbtensor:
            return single_channel_to_rgb_tensor(self.lesion_seg)

        return self.lesion_seg


def paste_blend(
    x_start: int,
    y_start: int,
    img,
    mask: np.ndarray,
    canvas,
) -> np.ndarray:
    if (img.shape[0] % 2) != 0:
        raise ValueError("Assumes even shape")

    if (img.shape[1]) % 2 != 0:
        raise ValueError("Assumes even shape")

    x_range_start = int(x_start - mask.shape[0] * 0.5)
    x_range_end = int(x_start + mask.shape[0] * 0.5)

    img_blend = canvas.copy()
    target_region = canvas[
        x_range_start:x_range_end,
        int(y_start - mask.shape[1] * 0.5) : int(y_start + mask.shape[1] * 0.5),
    ]
    img_blend[
        x_range_start:x_range_end,
        int(y_start - mask.shape[1] * 0.5) : int(y_start + mask.shape[1] * 0.5),
    ] = (mask * img) + (~mask * target_region)

    return img_blend


def blend_gradients(background_img, foreground_img, mask, gpu_id=0):
    """Blends the gradients of two images based on the mask.

    For proper gradient blending at the boundaries of the `mask`,
    the foreground image should have the textures of the original
    lesion that extend beyond the mask.

    Otherwise, there will can be a high gradient at the boundaries
    of the mask based on the color differences.

    Code borrowed from: https://github.com/owenzlz/DeepImageBlending/blob/94e6a1fb5a9a4d78ee07cf6b85431cf3bcdfb158/utils.py#L61

    Args:
        img (np.ndarray): H x W x 3 background image.
        dilated_paste_img (np.ndarray): H x W x 3 foreground image.
            The foreground object within the `mask` should have
            the original textures surrounding the `mask`.
        mask (np.ndarray): H x W binary mask, where a value of
            1 indicates the foreground and 0 indicates the background.

    Returns:
        list of tenors: A list of tensors of the blended grandients.
            The list contains three tensors representing the RGB gradients.
            Each tensor is of shape H x W.
    """

    paste_tensor = numpy2tensor(foreground_img, gpu_id = device)
    img_tensor = numpy2tensor(background_img, gpu_id = device)

    img_gradient = laplacian_filter_tensor(img_tensor, gpu_id=device)
    img_r_grad = img_gradient[0].squeeze().cpu().detach().numpy()
    img_g_grad = img_gradient[1].squeeze().cpu().detach().numpy()
    img_b_grad = img_gradient[2].squeeze().cpu().detach().numpy()

    dilated_gradient = laplacian_filter_tensor(paste_tensor, gpu_id=device)
    dilated_r_grad = dilated_gradient[0].squeeze().cpu().detach().numpy()
    dilated_g_grad = dilated_gradient[1].squeeze().cpu().detach().numpy()
    dilated_b_grad = dilated_gradient[2].squeeze().cpu().detach().numpy()

    r_grad_mod = dilated_r_grad * mask + img_r_grad * (1 - mask)
    g_grad_mod = dilated_g_grad * mask + img_g_grad * (1 - mask)
    b_grad_mod = dilated_b_grad * mask + img_b_grad * (1 - mask)
    rgb_gradient = [
        numpy2tensor(r_grad_mod, gpu_id = device),
        numpy2tensor(g_grad_mod, gpu_id = device),
        numpy2tensor(b_grad_mod, gpu_id = device),
    ]

    return rgb_gradient


def composite_image(input_img, canvas_mask, target_img):
    return canvas_mask * input_img + (1 - canvas_mask) * target_img


def render_views_with_textures(
    mesh_renderer,
    texture_image,
    texture_mask,
    pasted_texture,
    dilated_texture,
    original_texture,
    lesion_mask_id,
    pad=10,
):
    # Blended image. We are learning this.
    mesh_renderer.set_texture_image(texture_image=texture_image)
    # False for tensor.
    blended_images = mesh_renderer.render_view(asnumpy=False)
    blended_img_tensor = blended_images[:, ..., :3].transpose(1, 3).transpose(2, 3)

    # Lesion mask.
    mesh_renderer.set_texture_image(texture_image=texture_mask[:, :, np.newaxis])
    mask2d = mesh_renderer.render_view(asnumpy=True, asRGB=True)
    # lesion_mask = mesh_renderer.lesion_mask(mask2d[:, :, 0], lesion_mask_id)
    lesion_mask = mask2d * mesh_renderer.body_mask()[:, :, np.newaxis]
    lesion_mask = (lesion_mask[:, :, 0] > 0.5) * 1
    lesion_mask_tensor = numpy2tensor(lesion_mask[:, :, np.newaxis], gpu_id=device)
    xmin, xmax, ymin, ymax = mask2boundingbox(lesion_mask > 0.5, pad=pad)

    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > lesion_mask.shape[0]:
        xmax = lesion_mask.shape[0]
    if ymax > lesion_mask.shape[1]:
        ymax = lesion_mask.shape[1]

    # Pasted image.
    mesh_renderer.set_texture_image(texture_image=pasted_texture)
    pasted_img = mesh_renderer.render_view(asnumpy=True, asRGB=True)
    pasted_img_tensor = numpy2tensor(pasted_img, gpu_id=device)

    # Dilated image.
    mesh_renderer.set_texture_image(texture_image=dilated_texture)
    dilated_img = mesh_renderer.render_view(asnumpy=True, asRGB=True)

    # Original image.
    mesh_renderer.set_texture_image(texture_image=original_texture)
    original_img = mesh_renderer.render_view(asnumpy=True, asRGB=True)
    original_img_tensor = numpy2tensor(original_img, gpu_id=device)

    # Composite foreground and background to make the blended image.
    composite_img_tensor = composite_image(
        blended_img_tensor, lesion_mask_tensor, original_img_tensor
    )

    # Gradient loss.
    gt_gradient = blend_gradients(
        original_img[xmin:xmax, ymin:ymax, :],
        dilated_img[xmin:xmax, ymin:ymax, :],
        lesion_mask[xmin:xmax, ymin:ymax],
    )

    composite = composite_img_tensor[:, :, xmin:xmax, ymin:ymax]
    original = original_img_tensor[:, :, xmin:xmax, ymin:ymax]
    mask = lesion_mask_tensor[:, :, xmin:xmax, ymin:ymax]
    pasted = pasted_img_tensor[:, :, xmin:xmax, ymin:ymax]
    return {
        "composite": composite,
        "original": original,
        "mask": mask,
        "pasted": pasted,
    }, gt_gradient
