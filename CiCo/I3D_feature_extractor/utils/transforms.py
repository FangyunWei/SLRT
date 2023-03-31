import random

import numpy as np
import torch

from .imutils import im_to_video, video_to_im


def bbox_format(bbox, src, dest):
    if src == "yxyx" and dest == "cenhw":
        hw = bbox[:, 2:] - bbox[:, :2]
        cen = bbox[:, :2] + (hw / 2)
        bbox = np.hstack((cen, hw))
    elif src == "cenhw" and dest == "yxyx":
        yx_min = bbox[:, :2] - (bbox[:, 2:] / 2)
        yx_max = bbox[:, :2] + (bbox[:, 2:] / 2)
        bbox = np.hstack((yx_min, yx_max))
    else:
        raise ValueError(f"Unsupported bbox conversion [{src}] -> [{dest}]")
    return bbox


def scale_yxyx_bbox(bbox_yxyx, scale):
    """Apply isotropic scaling factors to a array of bounding boxes.

    Args:
        bbox_yxyx (np.ndarray): An (N x 4) array of N bounding boxes in the format
            `ymin,xmin,ymax,xmax`.
        scale (np.ndarray): An (N x 1) array of floats, to be applied multiplicatively
            to the widths and heights of each box.

    Returns:
        (np.ndarray): An (N x 4) array of N scaled bounding boxes.
    """
    bbox_cenhw = bbox_format(bbox_yxyx, src="yxyx", dest="cenhw")
    bbox_cenhw[:, 2:] = bbox_cenhw[:, 2:] * scale
    return bbox_format(bbox_cenhw, src="cenhw", dest="yxyx")


def color_normalize(x, mean, std):
    """Normalize a tensor of images by subtracting (resp. dividing) by the mean (resp.
    std. deviation) statistics of a dataset in RGB space.
    """
    if x.dim() in {3, 4}:
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        assert x.size(0) == 3, "For single video format, expected RGB along first dim"
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
    elif x.dim() == 5:
        assert (
            x.shape[1] == 3
        ), "For batched video format, expected RGB along second dim"
        x[:, 0].sub_(mean[0]).div_(std[0])
        x[:, 1].sub_(mean[1]).div_(std[1])
        x[:, 2].sub_(mean[2]).div_(std[2])
    return x


def im_color_jitter(rgb, num_in_frames=1, thr=0.2, deterministic_jitter_val=None):
    """Apply color jittering to a tensor of image frames by perturbing in RGB space.

    Args:
        `rgb` (torch.Tensor[float32]): A tensor of input images, which can be in one of
            two supported formats:
                3 dimensional input (3x<num_in_frames>) x H x W
                5 dimensional tensors: B x 3 x <num_in_frames> x H x W
        `num_in_frames` (int): the number of frames per "clip".
        `thr` (float): the magnitude of the jitter to be applied
        `deterministic_jitter_val` (list :: None): if supplied, use the given list of
            three (floating point) values to select the magnitude of the jitter to be
            applied to the R, G and B channels.

    Returns:
        (torch.Tensor[float32]): A jittered tensor (with the same shape as the input)
    """
    assert rgb.dim() in {3, 5}, "only 3 or 5 dim tensors are supported"
    supported_types = (torch.FloatTensor, torch.cuda.FloatTensor)
    assert isinstance(rgb, supported_types), "expected single precision inputs"
    if rgb.min() < 0:
        print(f"Warning: rgb.min() {rgb.min()} is less than 0.")
    if rgb.max() > 1:
        print(f"Warning: rgb.max() {rgb.max()} is more than 1.")
    if deterministic_jitter_val:
        assert (
            len(deterministic_jitter_val) == 3
        ), "expected to be provided 3 fixed vals"
        rjitter, gjitter, bjitter = deterministic_jitter_val
    else:
        rjitter = random.uniform(1 - thr, 1 + thr)
        gjitter = random.uniform(1 - thr, 1 + thr)
        bjitter = random.uniform(1 - thr, 1 + thr)
    if rgb.dim() == 3:
        rgb = im_to_video(rgb)
        assert (
            rgb.shape[1] == num_in_frames
        ), "Unexpected number of input frames per clip"
        rgb[0, :, :, :].mul_(rjitter).clamp_(0, 1)
        rgb[1, :, :, :].mul_(gjitter).clamp_(0, 1)
        rgb[2, :, :, :].mul_(bjitter).clamp_(0, 1)
        rgb = video_to_im(rgb)
    elif rgb.dim() == 5:
        assert rgb.shape[1] == 3, "expecte RGB to lie on second axis"
        assert (
            rgb.shape[2] == num_in_frames
        ), "Unexpected number of input frames per clip"
        rgb[:, 0].mul_(rjitter).clamp_(0, 1)
        rgb[:, 1].mul_(gjitter).clamp_(0, 1)
        rgb[:, 2].mul_(bjitter).clamp_(0, 1)
    return rgb
