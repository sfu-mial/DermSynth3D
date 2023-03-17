import random
import cv2
import albumentations as A
from dermsynth3d.utils.colorconstancy import shade_of_gray_cc


class ColorConstancyGray(A.core.transforms_interface.ImageOnlyTransform):
    def __init__(
        self,
        always_apply=False,
        p=0.5,
    ):
        super(ColorConstancyGray, self).__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return shade_of_gray_cc(img)

    def get_transform_init_args_names(self):
        return ()


class ResizeInterpolate(A.core.transforms_interface.DualTransform):
    """
    Resize the input to the given height and width.

    Args:
        height (int): desired height of the output.
        width (int): desired width of the output.
        interpolation (OpenCV flag): flag that is used to specify
            the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,
            cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        height,
        width,
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=1,
    ):
        super(ResizeInterpolate, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img, interpolation=None, **params):
        interpolation = self.interpolation

        interpolations = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
        ]
        # cv2.INTER_CUBIC,  cv2.INTER_LANCZOS4]
        if self.interpolation is None:
            interpolation = random.choice(interpolations)

        return A.augmentations.geometric.functional.resize(
            img, height=self.height, width=self.width, interpolation=interpolation
        )

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]
        scale_x = self.width / width
        scale_y = self.height / height
        return A.augmentations.geometric.functional.keypoint_scale(
            keypoint, scale_x, scale_y
        )

    def get_transform_init_args_names(self):
        return ("height", "width", "interpolation")
