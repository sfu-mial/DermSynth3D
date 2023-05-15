import numpy as np
from scipy import ndimage

from dermsynth3d.utils.tensor import max_value_in_window


class UVViewMapper:
    def __init__(
        self,
        view_uvs,
        paste_img,
        body_mask_view,
        lesion_mask_view,
        texture_img_size: int,
    ):
        """Maps the 2D view to a 2D texture image.

        Args:
            view_uvs (np.ndarray): H x W x 2 array of floats of the view.
                Each element is a UV coordinate in the range of [0, 1].
            paste_img (np.ndarray): H x W x 3 image of the view with the
                pasted lesion. Each element is an RGB pixel.
            body_mask_view (np.ndarray): H x W binary image where
                True indicates the body foreground, and
                False indicates the background.
            lesion_mask_view (np.ndarray): H x W binary image where
                True indicates the lesion foreground, and False otherwise.
            texture_img_size (int): Integer size of the texture image,
                which will be of size `img_size` x `img_size`.
        """

        self.view_uvs = view_uvs
        self.texture_img_size = texture_img_size

        teximage, lesion_texture_mask = uv_view_to_texture_image_mask(
            uv_view=self.view_uvs,
            image_view=paste_img,
            mask_view=body_mask_view,
            mask_seg=lesion_mask_view,
            img_size=texture_img_size,
        )

        self.teximage = teximage
        self.lesion_texture_mask = lesion_texture_mask

        self.max_dist_view = max_window_distance(view_uvs)
        self.max_body_dist = self.max_dist_view * body_mask_view

        self.padder = TexturePadding(
            lesion_texture_mask=self.lesion_texture_mask, texture_image=self.teximage
        )

    def max_body_distance(self, dist_thresh=0.1):
        body_seam_uvs = self.view_uvs[self.max_body_dist > dist_thresh]
        body_seam_uvs_image = uv_map_to_pixels(body_seam_uvs, self.texture_img_size)

        return body_seam_uvs_image

    def body_seams_xy(self, dist_thresh=0.1):
        body_seam_uvs_image = self.max_body_distance(dist_thresh)
        body_seams_x = self.texture_img_size - body_seam_uvs_image[:, 1]
        body_seams_y = body_seam_uvs_image[:, 0]

        return body_seams_x, body_seams_y

    def texture_pad(self, seam_thresh=0.1, niter=1):
        body_seams_x, body_seams_y = self.body_seams_xy(dist_thresh=seam_thresh)

        mask_pad, tex_pad = self.padder.texture_pad(
            body_seams_x, body_seams_y, niter=niter
        )

        return mask_pad, tex_pad


class TexturePadding:
    def __init__(self, lesion_texture_mask, texture_image):
        self.LESION_CHANNEL = 1
        self.PAD_CHANNEL = 2
        self.LESION_PAD_CHANNEL = 0

        self.lesion_texture_mask = lesion_texture_mask
        self.img_size = self.lesion_texture_mask.shape[0]

        assert (
            self.img_size == self.lesion_texture_mask.shape[1]
        ), "Expects equal dimensions."

        self.texture_image = texture_image

    def texture_pad(self, seams_x, seams_y, niter=1):
        padded_textures = np.zeros(
            shape=(self.img_size, self.img_size, 3), dtype=np.float32
        )
        padded_textures[:, :, self.LESION_CHANNEL] = self.lesion_texture_mask

        mask_padded, texture_padded = self.body_lesion_aware_seam_padding(
            seams_x, seams_y, padded_textures, self.texture_image
        )

        if niter > 1:
            for idx in np.arange(niter - 1):
                seams_x, seams_y = np.where(
                    mask_padded[:, :, self.PAD_CHANNEL].squeeze() > 0
                )
                mask_padded, texture_padded = self.body_lesion_aware_seam_padding(
                    seams_x, seams_y, mask_padded, texture_padded
                )

        return mask_padded, texture_padded

    def padded_lesion_mask(self, mask_padded):
        """Combine the lesion and lesion padded into a binary array.

        Non-zero values at the lesion or lesion padding channels
        are returned as True.

        Args:
            mask_padded (np.ndarray): H x W x C array where non-zero values
                indicate a value to include in a mask.
                The channels `C` correspond to the channels
                as defined in the __init__ of this class.

        Returns:
            np.ndarray: H x W x 1 binary array
        """
        lesion_pad_channel = mask_padded[:, :, self.LESION_PAD_CHANNEL]
        lesion_channel = mask_padded[:, :, self.LESION_CHANNEL]
        # Binary image where True indicates lesion or lesion padding.
        lesion_padded_mask = (lesion_pad_channel + lesion_channel) > 0

        return lesion_padded_mask[:, :, np.newaxis]

    def body_lesion_aware_seam_padding(
        self, seams_x, seams_y, teximage_seams, teximage
    ):
        """
        Pad around seams of the texture map to obtain uniform blending.
        """
        seam_map = teximage_seams.copy()
        texture_padded = teximage.copy()

        for x, y in zip(seams_x, seams_y):
            # Get the lesion mask of the patch.
            mask_patch = teximage_seams[
                (x - 1) : (x + 2), (y - 1) : (y + 2), self.LESION_CHANNEL
            ]

            pad_patch = teximage_seams[
                (x - 1) : (x + 2), (y - 1) : (y + 2), self.PAD_CHANNEL
            ]

            # Get the patch of the corresponding image.
            img_patch = teximage[(x - 1) : (x + 2), (y - 1) : (y + 2), :]

            isrgb_patch = img_patch.sum(axis=2) > 0

            # Assign to the pad_channel, the mask padding for the lesion.
            seam_map[(x - 1) : (x + 2), (y - 1) : (y + 2), self.PAD_CHANNEL] = (
                (1 - isrgb_patch) * (1 - pad_patch)
            ) + pad_patch

            lesion_pad_patch = teximage_seams[
                (x - 1) : (x + 2), (y - 1) : (y + 2), self.LESION_PAD_CHANNEL
            ]

            if lesion_pad_patch[1, 1] > 0:
                seam_map[
                    (x - 1) : (x + 2), (y - 1) : (y + 2), self.LESION_PAD_CHANNEL
                ] = ((1 - pad_patch) * (1 - lesion_pad_patch) + lesion_pad_patch) * (
                    1 - mask_patch
                ) * (
                    (1 - isrgb_patch) * (1 - lesion_pad_patch)
                ) + lesion_pad_patch
            elif mask_patch[1, 1] > 0:
                seam_map[
                    (x - 1) : (x + 2), (y - 1) : (y + 2), self.LESION_PAD_CHANNEL
                ] = (1 - mask_patch) * (1 - isrgb_patch)

            # If there are some zero rgb values in the patch,
            # this indicates a boundary pixel that should be filled.
            img_patch_pad = texture_padded[(x - 1) : (x + 2), (y - 1) : (y + 2), :]
            if np.sum(img_patch_pad == 0) > 0:
                # Get the RGB components.
                r = img_patch_pad[:, :, 0]
                g = img_patch_pad[:, :, 1]
                b = img_patch_pad[:, :, 2]
                # In the 3x3 window, for any zeros, assign the mean non-zero value.
                r[r == 0] = r[r > 0].mean()
                g[g == 0] = g[g > 0].mean()
                b[b == 0] = b[b > 0].mean()

        return seam_map, texture_padded


def uv_map_to_pixels(uv_map, img_size):
    uvs_image = uv_map * img_size

    u_ceil = np.ceil(uvs_image[:, 0, np.newaxis]).astype(np.int)
    v_ceil = np.ceil(uvs_image[:, 1, np.newaxis]).astype(np.int)
    u_floor = np.floor(uvs_image[:, 0, np.newaxis]).astype(np.int)
    v_floor = np.floor(uvs_image[:, 1, np.newaxis]).astype(np.int)

    uvs_ceil_ceil = np.concatenate((u_ceil, v_ceil), axis=1)
    uvs_floor_floor = np.concatenate((u_floor, v_floor), axis=1)
    uvs_ceil_floor = np.concatenate((u_ceil, v_floor), axis=1)
    uvs_floor_ceil = np.concatenate((u_floor, v_ceil), axis=1)

    image_uvs = np.concatenate(
        (uvs_ceil_ceil, uvs_floor_floor, uvs_ceil_floor, uvs_floor_ceil)
    )
    return image_uvs


def uvimage2pixels(uv_image, img_size=(4096, 4096)):
    """
    Convert uv space into texture image coordinates.
    """
    # Convert uv space to image pixel space.
    uv_pixels = uv_image * img_size

    # Discrete locations the uv can map to in pixel space.
    u_ceil = np.ceil(uv_pixels[:, :, 0, np.newaxis]).astype(np.int)
    v_ceil = np.ceil(uv_pixels[:, :, 1, np.newaxis]).astype(np.int)
    u_floor = np.floor(uv_pixels[:, :, 0, np.newaxis]).astype(np.int)
    v_floor = np.floor(uv_pixels[:, :, 1, np.newaxis]).astype(np.int)

    uvs_ceil_ceil = np.concatenate((u_ceil, v_ceil), axis=-1)
    uvs_floor_floor = np.concatenate((u_floor, v_floor), axis=-1)
    uvs_ceil_floor = np.concatenate((u_ceil, v_floor), axis=-1)
    uvs_floor_ceil = np.concatenate((u_floor, v_ceil), axis=-1)

    uv_discrete_pixels = np.concatenate(
        (
            uvs_ceil_ceil[np.newaxis,],
            uvs_floor_floor[np.newaxis,],
            uvs_ceil_floor[np.newaxis,],
            uvs_floor_ceil[np.newaxis,],
        )
    )

    return uv_discrete_pixels


def uv_view_to_texture_image_mask(
    uv_view, image_view, mask_view, mask_seg, img_size: int
):
    # Increase the resolution for better mapping.
    # Use nearest-neighbors for the UVs since we do not want to interpolate.
    uv_view_highres = ndimage.zoom(uv_view, (2, 2, 1), order=0)

    # Use interpolation for the image.
    image_view_highres = ndimage.zoom(image_view, (2, 2, 1), order=1)

    # Nearest neighbors for the masks.
    mask = ndimage.zoom(mask_view, (2, 2), order=0)
    seg = ndimage.zoom(mask_seg, (2, 2), order=0)

    # Continous UVs to pixel space.
    uv_pixel_coords = uvimage2pixels(uv_view_highres)

    teximage = np.zeros(shape=(img_size, img_size, 3), dtype=np.float32)
    lesion_texture_mask = np.zeros(shape=(img_size, img_size), dtype=np.int64)

    for uv_discrete in uv_pixel_coords:
        # Skip over pixels where the face indicates is the background.
        teximage[
            (img_size - uv_discrete[:, :, 1][mask > 0], uv_discrete[:, :, 0][mask > 0])
        ] = image_view_highres[mask > 0]
        # Skip over pixels that are not part of the lesion segmentation.
        # i.e., only put pixels for lesion.
        lesion_texture_mask[
            (img_size - uv_discrete[:, :, 1][seg > 0], uv_discrete[:, :, 0][seg > 0])
        ] = 1

    return teximage, lesion_texture_mask


def max_window_distance(view_uvs):
    max_bary_window0 = max_value_in_window(view_uvs[:, :, 0], (3, 3)).squeeze()
    max_bary_window1 = max_value_in_window(view_uvs[:, :, 1], (3, 3)).squeeze()
    view_max_dist = np.sqrt(
        (max_bary_window0 - view_uvs[:, :, 0]) ** 2
        + (max_bary_window1 - view_uvs[:, :, 1]) ** 2
    )
    return view_max_dist
