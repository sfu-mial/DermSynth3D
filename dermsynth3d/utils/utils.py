import random
import cv2
import sys
import yaml
import logging
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import datetime
import time
from collections import defaultdict, deque
import sys

from dermsynth3d.utils.channels import Target


def random_offset(a=-0.025, b=0.025):
    return random_bound(a, b)


def random_bound(lower, upper):
    r = np.random.random_sample()
    return (upper - lower) * r + lower


def mask2boundingbox(mask, pad=0):
    xx, yy = np.where(mask)
    xmin = xx.min() - pad
    xmax = xx.max() + pad
    ymin = yy.min() - pad
    ymax = yy.max() + pad
    return xmin, xmax, ymin, ymax


def blend_background(fore_img, back_img, fore_mask, soft_blend=False):
    """
    Blend background to the foreground
    """
    background = back_img * (~(fore_mask[:, :, np.newaxis]))
    blend_img = fore_img * fore_mask[:, :, np.newaxis] + background

    if soft_blend:
        soft_mask = cv2.GaussianBlur(fore_mask * 1.0, (5, 5), cv2.BORDER_DEFAULT)
        soft_mask[soft_mask > 0.999] = 1
        blend_img = (blend_img * soft_mask[:, :, np.newaxis]) + (
            1 - soft_mask[:, :, np.newaxis]
        ) * back_img

    return blend_img


def skin_mask_no_lesion(skin_mask: np.ndarray, lesion_mask: np.ndarray):
    assert skin_mask.dtype == "bool"
    assert lesion_mask.dtype == "bool"
    mask = skin_mask.copy()
    mask = skin_mask != lesion_mask
    return mask


def make_masks(lesion_mask, skin_mask):
    # For segmentation only 3 channels: Lesion, Skin and Non-skin
    n_channels = 3
    masks = np.zeros(
        shape=(lesion_mask.shape[0], lesion_mask.shape[1], n_channels),
        dtype=np.float32,
    )

    masks[:, :, Target.LESION] = lesion_mask
    masks[:, :, Target.SKIN] = np.asarray(
        skin_mask_no_lesion(skin_mask, lesion_mask), dtype=np.float32
    )
    masks[:, :, Target.NONSKIN] = np.asarray(~skin_mask, dtype=np.float32)

    return masks


def random_resize_crop_seg_lesion(
    seg_crop,
    lesion_crop,
    min_scale=2,
    max_scale=None,
    min_crop_size=10,
    maintain_aspect=True,
):
    """
    Randomly resize Lesion to paste
    """
    seg_crop_pil = Image.fromarray(seg_crop)
    lesion_crop_pil = Image.fromarray(lesion_crop)

    if max_scale is None:
        max_size = np.asarray(seg_crop_pil.size)
    else:
        max_size = np.asarray(seg_crop_pil.size) // max_scale

    if min_scale <= max_scale:
        print(max_scale)
        min_scale = max_scale

    min_size = np.asarray(seg_crop_pil.size) // min_scale

    if min_size[0] == max_size[0]:
        min_size[0] = min_size[0] - 1

    if min_size[1] == max_size[1]:
        min_size[1] = min_size[1] - 1

    resize_dim_0 = random.randrange(min_size[0], max_size[0])
    if maintain_aspect:
        wpercent = resize_dim_0 / float(seg_crop_pil.size[0])
        resize_dim_1 = int((float(seg_crop_pil.size[1]) * float(wpercent)))
    else:
        resize_dim_1 = random.randrange(min_size[1], max_size[1])

    # Make sure the dimensions have an even size
    # (odd gives errors with pasting)
    if resize_dim_0 % 2 != 0:
        resize_dim_0 -= 1

    if resize_dim_1 % 2 != 0:
        resize_dim_1 -= 1
    if (resize_dim_0 < min_crop_size) or (resize_dim_1 < min_crop_size):
        return None, None

    resize = (resize_dim_0, resize_dim_1)
    seg_crop_pil = seg_crop_pil.resize(size=resize, resample=Image.NEAREST)
    lesion_crop_pil = lesion_crop_pil.resize(size=resize, resample=Image.NEAREST)
    resized_lesion = np.asarray(lesion_crop_pil).astype(np.float32) / 255
    resized_crop = np.asarray(seg_crop_pil).astype(np.float32) / 255
    resized_crop = resized_crop > 0

    if resized_lesion.shape[0] % 2 != 0:
        raise ValueError("Error: output an even shape")

    if resized_lesion.shape[1] % 2 != 0:
        raise ValueError("Error: output an even shape")

    return resized_lesion, resized_crop


def pix_face_in_set(pix_to_face, face_indexes):
    pix_in_set = np.zeros_like(pix_to_face[:, :])
    for x in np.arange(pix_in_set.shape[0]):
        for y in np.arange(pix_in_set.shape[1]):
            if len(pix_to_face.shape) > 2:
                pix_face = set(pix_to_face[x, y])
            else:
                pix_face = set([pix_to_face[x, y]])

            if pix_face & face_indexes:
                pix_in_set[x, y] = 1

    return pix_in_set > 0


def yaml_loader(path):
    stream = open(path)
    file = yaml.safe_load(stream)
    return file


def get_logger(filename):
    # Logging configuration: set the basic configuration of the logging system
    log_formatter = logging.Formatter(
        fmt="%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s",
        datefmt="%m-%d %H:%M",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler("{}.log".format(filename))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # Stderr logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.DEBUG)
    logger.addHandler(std_handler)
    return logger


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
