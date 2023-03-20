import random
import numbers
import math
import collections
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import ImageOps, Image, ImageFilter
import numpy as np


class Padding:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, img):
        return ImageOps.expand(img, border=self.pad, fill=0)


class Scale:
    def __init__(self, size, interpolation=Image.BICUBIC):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        img1 = imgmap[0]
        if isinstance(self.size, int):
            w, h = img1.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return imgmap
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
        else:
            return [i.resize(self.size, self.interpolation) for i in imgmap]


class CenterCrop:
    def __init__(self, size, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]


class FiveCrop:
    def __init__(self, size, where=1):
        # 1=topleft, 2=topright, 3=botleft, 4=botright, 5=center
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.where = where 

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        th, tw = self.size
        if (th > h) or (tw > w):
            raise ValueError("Requested crop size {} is bigger than input size {}".format(self.size, (h,w)))
        if self.where == 1:
            return [i.crop((0, 0, tw, th)) for i in imgmap]
        elif self.where == 2:
            return [i.crop((w-tw, 0, w, th)) for i in imgmap]
        elif self.where == 3:
            return [i.crop((0, h-th, tw, h)) for i in imgmap]
        elif self.where == 4:
            return [i.crop((w-tw, h-tw, w, h)) for i in imgmap]
        elif self.where == 5:
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))
            return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]


class RandomSizedCrop:
    def __init__(self, size, 
        interpolation=Image.BICUBIC, 
        consistent=True, 
        p=1.0, 
        seq_len=0, 
        bottom_area=0.2,
        aspect_ratio_min=3./4, aspect_ratio_max=4./3,
        center_crop=True, center_crop_size=224):
        self.size = size
        self.interpolation = interpolation
        self.consistent = consistent
        self.threshold = p 
        self.seq_len = seq_len
        self.bottom_area = bottom_area
        self.aspect_ratio_min = aspect_ratio_min
        self.aspect_ratio_max = aspect_ratio_max
        self.center_crop=center_crop
        self.center_crop_size=center_crop_size

        # print('threshold', self.threshold)
        # print('bottom_area', self.bottom_area)
        # print('aspect_ratio', self.aspect_ratio_min, self.aspect_ratio_max)
        # print('center crop', self.center_crop)
        # print('center crop', self.center_crop_size)

    def __call__(self, imgmap):
        img1 = imgmap[0]
        if random.random() < self.threshold: # do RandomSizedCrop
            for attempt in range(10):
                area = img1.size[0] * img1.size[1]
                target_area = random.uniform(self.bottom_area, 1) * area
                aspect_ratio = random.uniform(self.aspect_ratio_min, self.aspect_ratio_max)

                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))

                if self.consistent:
                    if random.random() < 0.5:
                        w, h = h, w
                    if w <= img1.size[0] and h <= img1.size[1]:
                        x1 = random.randint(0, img1.size[0] - w)
                        y1 = random.randint(0, img1.size[1] - h)

                        imgmap = [i.crop((x1, y1, x1 + w, y1 + h)) for i in imgmap]
                        for i in imgmap: assert(i.size == (w, h))

                        return [i.resize((self.size, self.size), self.interpolation) for i in imgmap]
                else:
                    result = []

                    if random.random() < 0.5:
                        w, h = h, w

                    for idx, i in enumerate(imgmap):
                        if w <= img1.size[0] and h <= img1.size[1]:
                            if idx % self.seq_len == 0:
                                x1 = random.randint(0, img1.size[0] - w)
                                y1 = random.randint(0, img1.size[1] - h)
                            result.append(i.crop((x1, y1, x1 + w, y1 + h)))
                            assert(result[-1].size == (w, h))
                        else:
                            result.append(i)

                    assert len(result) == len(imgmap)
                    return [i.resize((self.size, self.size), self.interpolation) for i in result] 

            # Fallback
            scale = Scale(self.size, interpolation=self.interpolation)
            crop = CenterCrop(self.size)
            return crop(scale(imgmap))
        else: #don't do RandomSizedCrop, do CenterCrop
            if self.center_crop:
                crop = CenterCrop(self.center_crop_size)
                return crop(imgmap)
            else:
                return imgmap


class RandomHorizontalFlip:
    def __init__(self, consistent=True, command=None, seq_len=0):
        self.consistent = consistent
        if seq_len != 0:
            self.consistent = False 
        if command == 'left':
            self.threshold = 0
        elif command == 'right':
            self.threshold = 1
        else:
            self.threshold = 0.5
        self.seq_len = seq_len
    def __call__(self, imgmap):
        if self.consistent:
            if random.random() < self.threshold:
                return [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for idx, i in enumerate(imgmap):
                if idx % self.seq_len == 0: th = random.random()
                if th < self.threshold:
                    result.append(i.transpose(Image.FLIP_LEFT_RIGHT))
                else:
                    result.append(i) 
            assert len(result) == len(imgmap)
            return result 


class RandomGray:
    '''Actually it is a channel splitting, not strictly grayscale images'''
    def __init__(self, consistent=True, p=0.5, dynamic=False, seq_len=0):
        self.consistent = consistent
        if seq_len != 0:
            self.consistent = False
        self.p = p # prob to grayscale
        self.seq_len = seq_len
    def __call__(self, imgmap):
        tmp_p = self.p 
        if self.consistent:
            if random.random() < tmp_p:
                return [self.grayscale(i) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            if self.seq_len == 0:
                for i in imgmap:
                    if random.random() < tmp_p:
                        result.append(self.grayscale(i))
                    else:
                        result.append(i)
            else:
                for idx, i in enumerate(imgmap):
                    if idx % self.seq_len == 0:
                        do_gray = random.random() < tmp_p
                    if do_gray: result.append(self.grayscale(i))
                    else: result.append(i)
            assert len(result) == len(imgmap)
            return result 

    def grayscale(self, img):
        channel = np.random.choice(3)
        np_img = np.array(img)[:,:,channel]
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
        return img 


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, consistent=False, p=1.0, seq_len=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.consistent = consistent
        self.threshold = p 
        self.seq_len = seq_len

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
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
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)

        return transform

    def __call__(self, imgmap):
        if random.random() < self.threshold: # do ColorJitter
            if self.consistent:
                transform = self.get_params(self.brightness, self.contrast,
                                            self.saturation, self.hue)
                return [transform(i) for i in imgmap]
            else:
                if self.seq_len == 0:
                    return [self.get_params(self.brightness, self.contrast, self.saturation, self.hue)(img) for img in imgmap]
                else:
                    result = []
                    for idx, img in enumerate(imgmap):
                        if idx % self.seq_len == 0:
                            transform = self.get_params(self.brightness, self.contrast,
                                                        self.saturation, self.hue)
                        result.append(transform(img))
                    return result

                # result = []
                # for img in imgmap:
                #     transform = self.get_params(self.brightness, self.contrast,
                #                                 self.saturation, self.hue)
                #     result.append(transform(img))
                # return result
        else: # don't do ColorJitter, do nothing
            return imgmap 

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RandomRotation:
    def __init__(self, consistent=True, degree=15, p=1.0):
        self.consistent = consistent
        self.degree = degree 
        self.threshold = p
    def __call__(self, imgmap):
        if random.random() < self.threshold: # do RandomRotation
            if self.consistent:
                deg = np.random.randint(-self.degree, self.degree, 1)[0]
                return [i.rotate(deg, expand=True) for i in imgmap]
            else:
                return [i.rotate(np.random.randint(-self.degree, self.degree, 1)[0], expand=True) for i in imgmap]
        else: # don't do RandomRotation, do nothing
            return imgmap 


class ToTensor:
    def __call__(self, imgmap):
        totensor = transforms.ToTensor()
        return [totensor(i) for i in imgmap]

class ToPIL:
    def __call__(self, imgmap):
        topil = transforms.ToPILImage()
        return [topil(i) for i in imgmap]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.], seq_len=0):
        self.sigma = sigma
        self.seq_len = seq_len

    def __call__(self, imgmap):
        result = []
        for idx, img in enumerate(imgmap):
            if idx % self.seq_len == 0:
                sigma = random.uniform(self.sigma[0], self.sigma[1])
            result.append(img.filter(ImageFilter.GaussianBlur(radius=sigma)))
        return result

class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    def __call__(self, imgmap):
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        return [normalize(i) for i in imgmap]


class TwoClipTransform:
    """Take two random transforms on two clips"""
    def __init__(self, base_transform, null_transform, seq_len, p=0.3):
        # p = probability to use base_transform
        self.base = base_transform
        self.null = null_transform
        self.p = p 
        self.seq_len = seq_len # channel to split the tensor into two

    def __call__(self, x):
        # target: list of image
        assert len(x) == 2 * self.seq_len

        if random.random() < self.p:
            tr1 = self.base
        else:
            tr1 = self.null

        if random.random() < self.p:
            tr2 = self.base
        else:
            tr2 = self.null

        q = tr1(x[0:self.seq_len])
        k = tr2(x[self.seq_len::])
        return q + k


class OneClipTransform:
    """Take two random transforms on one clips"""
    def __init__(self, base_transform, null_transform, seq_len):
        self.base = base_transform
        self.null = null_transform
        self.seq_len = seq_len # channel to split the tensor into two

    def __call__(self, x):
        # target: list of image
        assert len(x) == 2 * self.seq_len

        if random.random() < 0.5:
            tr1, tr2 = self.base, self.null
        else:
            tr1, tr2 = self.null, self.base

        # randomly abandon half
        if random.random() < 0.5:
            xx = x[0:self.seq_len]
        else:
            xx = x[self.seq_len::]

        q = tr1(xx)
        k = tr2(xx)
        return q + k


class TransformController:
    def __init__(self, transform_list, weights):
        self.transform_list = transform_list
        self.weights = weights
        self.num_transform = len(transform_list)
        assert self.num_transform == len(self.weights)

    def __call__(self, x):
        idx = random.choices(range(self.num_transform), weights=self.weights)[0]
        return self.transform_list[idx](x)

    def __str__(self):
        string = 'TransformController: %s with weights: %s' % (str(self.transform_list), str(self.weights))
        return string



class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

