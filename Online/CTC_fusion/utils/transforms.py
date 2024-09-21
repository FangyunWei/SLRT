# from references/video_classification
# and torchvision/transforms/functional_tensor.py

# Support data argumentation for 4D tensor, [N, H, W, C]

import torch
import random
import numbers
import torchvision
import numpy as np 
import math

def crop(vid, i, j, h, w):
    return vid[..., i:(i + h), j:(j + w)]


def center_crop(vid, output_size):
    h, w = vid.shape[-2:]
    th, tw = output_size

    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(vid, i, j, th, tw)


def hflip(vid):
    return vid.flip(dims=(-1,))


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False)


def pad(vid, padding, fill=0, padding_mode="constant"):
    # NOTE: don't want to pad on temporal dimension, so let as non-batch
    # (4d) before padding. This works as expected
    return torch.nn.functional.pad(vid, padding, value=fill, mode=padding_mode)

def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255
    # [N,H,W,C] -> [C,N,H,W]

def to_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32)
    # [N,H,W,C] -> [C,N,H,W]

def normalize(vid, mean, std, channel=0):
    # shape = (-1,) + (1,) * (vid.dim() - 1)
    assert vid.size(channel) == 3
    shape = [1] * vid.dim(); shape[channel] = -1
    mean = torch.as_tensor(mean).to(vid.device).reshape(shape)
    std = torch.as_tensor(std).to(vid.device).reshape(shape)
    return (vid - mean) / std

def normalize_all_channel(vid, mean, std):
    return (vid - mean) / std

def rgb_to_grayscale(vid, channel=0):
    """Convert the given RGB Image Tensor to Grayscale.
    For RGB to Grayscale conversion, ITU-R 601-2 luma transform is performed which
    is L = R * 0.2989 + G * 0.5870 + B * 0.1140
    Args:
        vid (Tensor): Image to be converted to Grayscale in the form [C, N, H, W].
        channel: color channel
    Returns:
        Tensor: Grayscale video tensor [C, N, H, W].
    """
    assert vid.size(channel) == 3

    return (0.2989 * vid.select(channel,0) + 0.5870 * vid.select(channel,1) + 0.1140 * vid.select(channel,2)).to(vid.dtype)

def random_grayscale(vid, factor, channel=1):
    N = vid.size(channel)
    gray_map = np.random.uniform(size=(N,)) < factor
    if gray_map.sum() == 0: return vid 
    shape = [1]*vid.dim(); shape[channel] = N
    gray_map = torch.tensor(gray_map).to(vid.device).float().view(shape)
    gray = rgb_to_grayscale(vid, channel).unsqueeze(channel)
    return gray*gray_map+vid*(1-gray_map)


def adjust_brightness(vid, brightness_factor, channel=1):
    """Adjust brightness of an RGB image.
    Args:
        vid (Tensor): Video to be adjusted [C,N,H,W].
        brightness_factor (1D float tensor):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
        channel: N channel
    Returns:
        Tensor: Brightness adjusted video.
    """
    N = brightness_factor.size(0)
    assert vid.size(channel) == N
    shape = [1] * vid.dim(); shape[channel] = N
    brightness_factor = brightness_factor.view(shape)

    return _blend(vid, 0, brightness_factor)


def adjust_contrast(vid, contrast_factor, channel=1, gray_channel=0):
    """Adjust contrast of an RGB image.
    Args:
        vid (Tensor): Video to be adjusted [C,N,H,W].
        contrast_factor (1D float tensor): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
        channel: N channel
        gray_channel: gray channel of vid
    Returns:
        Tensor: Contrast adjusted video.
    """
    N = contrast_factor.size(0)
    assert vid.size(channel) == N 
    shape = [1] * vid.dim(); shape[channel] = N
    contrast_factor = contrast_factor.view(shape)
    # mean = rgb_to_grayscale(vid).to(torch.float).mean(1, keepdim=True).mean(2, keepdim=True)
    # mean: [N,1,1]
    mean_gray = rgb_to_grayscale(vid, gray_channel).to(torch.float) # not mean yet
    index = [i for i in range(vid.dim()-1)][::-1]
    if gray_channel < channel: 
        index.remove(channel-1)
    else:
        index.remove(channel)
    for i in index:
        mean_gray = mean_gray.mean(i)
    mean_gray = mean_gray.view(shape)
    return _blend(vid, mean_gray, contrast_factor)


def adjust_saturation(vid, saturation_factor, channel=1, gray_channel=0):
    """Adjust color saturation of an RGB image.
    Args:
        vid (Tensor): Video to be adjusted [C,N,H,W].
        saturation_factor (1D float tensor):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
        channel: N channel
        gray_channel: gray channel of vid
    Returns:
        Tensor: Saturation adjusted video.
    """
    N = saturation_factor.size(0)
    assert vid.size(channel) == N
    shape = [1] * vid.dim(); shape[channel] = N
    saturation_factor = saturation_factor.view(shape)

    return _blend(vid, rgb_to_grayscale(vid, gray_channel).unsqueeze(gray_channel), saturation_factor)


def _blend(img1, img2, ratio):
    bound = 1 if img1.dtype.is_floating_point else 255
    ratio = ratio.to(img1.dtype)
    return (ratio * img1 + (1 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def random_adjust_brightness(vid, brightness_factor, consistent, channel=1):
    N = vid.size(channel)
    if consistent:
        brightness_factor = np.array([random.uniform(brightness_factor[0], brightness_factor[1])]*N)
    else:
        brightness_factor = np.random.uniform(brightness_factor[0], brightness_factor[1], size=(N,))
    brightness_factor = torch.from_numpy(brightness_factor).to(vid.device)
    return adjust_brightness(vid, brightness_factor, channel)

def random_adjust_contrast(vid, contrast_factor, consistent, channel=1, gray_channel=0):
    N = vid.size(channel)
    if consistent:
        contrast_factor = np.array([random.uniform(contrast_factor[0], contrast_factor[1])]*N)
    else:
        contrast_factor = np.random.uniform(contrast_factor[0], contrast_factor[1], size=(N,))
    contrast_factor = torch.from_numpy(contrast_factor).to(vid.device)
    return adjust_contrast(vid, contrast_factor, channel, gray_channel) 

def random_adjust_saturation(vid, saturation_factor, consistent, channel=1, gray_channel=0):
    N = vid.size(channel)
    if consistent:
        saturation_factor = np.array([random.uniform(saturation_factor[0], saturation_factor[1])]*N)
    else:
        saturation_factor = np.random.uniform(saturation_factor[0], saturation_factor[1], size=(N,))
    saturation_factor = torch.from_numpy(saturation_factor).to(vid.device)
    return adjust_saturation(vid, saturation_factor, channel, gray_channel) 


# Class interface

class Stack:
    def __init__(self, dim=1):
        self.dim = dim
    def __call__(self, imgmap):
        return torch.stack(imgmap, self.dim)

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(vid, output_size):
        """Get parameters for ``crop`` for a random crop.
        """
        h, w = vid.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, vid):
        i, j, h, w = self.get_params(vid, self.size)
        return crop(vid, i, j, h, w)

class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(vid, output_size):
        """Get parameters for ``crop`` for a random sized crop.
        """
        for attempt in range(10):
            h, w = vid.shape[-2:]
            area = h*w 
            target_area = random.uniform(0.5, 1) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)
            tw = int(round(math.sqrt(target_area * aspect_ratio)))
            th = int(round(math.sqrt(target_area / aspect_ratio)))
            if tw <= w and th <= h:
                i = random.randint(0, h - th)
                j = random.randint(0, w - tw)
                return i, j, th, tw
        th, tw = output_size
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, vid):
        i, j, h, w = self.get_params(vid, self.size)
        return resize(crop(vid, i, j, h, w), self.size)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return center_crop(vid, self.size)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class ToFloatTensor(object):
    def __call__(self, vid):
        return to_float_tensor(vid)


class Normalize(object):
    def __init__(self, mean, std, channel=0):
        self.mean = mean
        self.std = std
        self.channel = channel

    def __call__(self, vid):
        return normalize(vid, self.mean, self.std, self.channel)

class Normalize_all_channel(object):
    def __init__(self, mean, std, channel=0):
        self.mean = mean # should be a float
        self.std = std
        self.channel = channel

    def __call__(self, vid):
        return normalize_all_channel(vid, self.mean, self.std)

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vid):
        if random.random() < self.p:
            return hflip(vid)
        return vid


class Pad(object):
    def __init__(self, padding, fill=0):
        self.padding = padding
        self.fill = fill

    def __call__(self, vid):
        return pad(vid, self.padding, self.fill)


class RandomGray(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vid):
        return random_grayscale(vid, self.p)


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, 
                 consistent=False, p=1.0, n_channel=1, gray_channel=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.consistent = consistent
        self.p = p
        self.n_channel = n_channel
        self.gray_channel = gray_channel

    def _check_input(self, value, name, center=1, bound=(0, float('inf'))):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, consistent, n_channel, gray_channel):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            transforms.append(torchvision.transforms.Lambda(lambda vid: 
                random_adjust_brightness(vid, brightness, consistent, n_channel)))

        if contrast is not None:
            transforms.append(torchvision.transforms.Lambda(lambda vid: 
                random_adjust_contrast(vid, contrast, consistent, n_channel, gray_channel)))

        if saturation is not None:
            transforms.append(torchvision.transforms.Lambda(lambda vid: 
                random_adjust_saturation(vid, saturation, consistent, n_channel, gray_channel)))

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)

        return transform

    def __call__(self, vid):
        if random.random() < self.p:
            transform = self.get_params(self.brightness, self.contrast, self.saturation, 
                                        self.consistent, self.n_channel, self.gray_channel)
            return transform(vid)
        return vid

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        return format_string

