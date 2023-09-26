#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
from numbers import Number
from typing import Sequence

import PIL.Image
import numpy as np

import mindspore
import mindspore.dataset.transforms
import mindspore.dataset.vision

from .context import x2ms_context
from .numpy import TensorNumpy
from .nn_functional import interpolate

_pil_interpolation_map = {
    PIL.Image.BILINEAR: mindspore.dataset.vision.Inter.BILINEAR,
    PIL.Image.NEAREST: mindspore.dataset.vision.Inter.NEAREST,
    PIL.Image.BICUBIC: mindspore.dataset.vision.Inter.BICUBIC,
}

_ms_np_type_map = {
    mindspore.float32: np.float32,
    mindspore.float64: np.float64,
    mindspore.float16: np.float16,
    mindspore.int64: np.int64,
    mindspore.int32: np.int32,
    mindspore.int8: np.int8,
    mindspore.uint8: np.uint8,
    mindspore.int16: np.int16,
}


class Resize(mindspore.dataset.vision.py_transforms.Resize):
    def __init__(self, size, interpolation=PIL.Image.BILINEAR, max_size=None, antialias=None):
        interpolation = _pil_interpolation_map.get(interpolation, mindspore.dataset.vision.Inter.BILINEAR)
        super().__init__(size=size, interpolation=interpolation)


class Compose(mindspore.dataset.py_transforms.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, *args):
        result = args[0]
        for transform in self.transforms:
            result = transform(result)
        if isinstance(result, np.ndarray):
            return mindspore.Tensor(result)
        return result


class RandomResizedCropAndInterpolation(mindspore.dataset.vision.py_transforms.RandomResizedCrop):
    interpolation_map = {
        'bilinear': mindspore.dataset.vision.Inter.BILINEAR,
        'nearest': mindspore.dataset.vision.Inter.NEAREST,
        'bicubic': mindspore.dataset.vision.Inter.BICUBIC,
        'antialias': mindspore.dataset.vision.Inter.ANTIALIAS
    }

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear'):
        interpolation = self.interpolation_map.get(interpolation, mindspore.dataset.vision.Inter.BILINEAR)
        super().__init__(size, scale, ratio, interpolation=interpolation)


class RandomHorizontalFlip(mindspore.dataset.vision.py_transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(prob=p)


class RandomVerticalFlip(mindspore.dataset.vision.py_transforms.RandomVerticalFlip):
    def __init__(self, p=0.5):
        super().__init__(prob=p)


class Normalize(mindspore.dataset.vision.py_transforms.Normalize):
    def __init__(self, mean, std, inplace=False):
        if isinstance(mean, mindspore.Tensor):
            mean = mean.asnumpy().tolist()
        if isinstance(std, mindspore.Tensor):
            std = std.asnumpy().tolist()
        if isinstance(mean, Number):
            mean = [mean]
        if isinstance(std, Number):
            std = [std]
        super().__init__(mean, std)

    def __call__(self, img):
        if x2ms_context.get_is_during_transform():
            return super().__call__(img)
        if isinstance(img, mindspore.Tensor):
            img = img.asnumpy()
        return mindspore.Tensor(super().__call__(img))


class ToTensor(mindspore.dataset.vision.py_transforms.ToTensor):
    def __call__(self, img):
        result = super().__call__(img)
        return TensorNumpy.create_tensor_numpy(result)


def to_tensor(pic):
    return TensorNumpy.create_tensor_numpy(ToTensor()(pic))


def resized_crop(img, top, left, height, width, size, interpolation):
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img


def crop(img, top: int, left: int, height: int, width: int):
    if not isinstance(img, mindspore.Tensor):
        return pil_crop(img, top, left, height, width)
    return tensor_crop(img, top, left, height, width)


def pil_crop(img, top: int, left: int, height: int, width: int):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    return img.crop((left, top, left + width, top + height))


def _is_pil_image(img) -> bool:
    return isinstance(img, PIL.Image.Image)


def _is_tensor_a_image(x) -> bool:
    return x.ndim >= 2


def tensor_crop(img, top, left, height, width):
    if not _is_tensor_a_image(img):
        raise TypeError("tensor is not a mindspore image.")
    return img[..., top:top + height, left:left + width]


def resize(img, size, interpolation=PIL.Image.BILINEAR):
    if not isinstance(img, mindspore.Tensor):
        return pil_resize(img, size=size, interpolation=interpolation)
    return tensor_resize(img, size=size, interpolation=interpolation)


def pil_resize(img, size, interpolation=PIL.Image.BILINEAR):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Sequence) and len(size) in (1, 2))):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int) or len(size) == 1:
        if isinstance(size, Sequence):
            size = size[0]
        width, height = img.size
        if (width <= height and width == size) or (height <= width and height == size):
            return img
        if width < height:
            ow = size
            oh = int(size * height / width)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * width / height)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def tensor_resize(img, size, interpolation=2):
    if not _is_tensor_a_image(img):
        raise TypeError("tensor is not a mindspore image.")
    if not isinstance(size, (int, tuple, list)):
        raise TypeError("Got inappropriate size arg")

    _interpolation_modes = {
        0: "nearest"
    }
    if interpolation not in _interpolation_modes:
        raise ValueError("This interpolation mode is unsupported with Tensor input")
    if isinstance(size, tuple):
        size = list(size)

    width, height = img.shape[-1], img.shape[-2]

    if isinstance(size, int):
        size_w, size_h = size, size
    elif len(size) < 2:
        size_w, size_h = size[0], size[0]
    else:
        size_w, size_h = size[1], size[0]  # Convention (h, w)

    if isinstance(size, int) or len(size) < 2:
        if width < height:
            size_h = int(size_w * height / width)
        else:
            size_w = int(size_h * width / height)

        if (width <= height and width == size_w) or (height <= width and height == size_h):
            return img

    # make image NCHW
    need_squeeze = False
    if img.ndim < 4:
        img = mindspore.ops.ExpandDims()(img, 0)
        need_squeeze = True

    mode = _interpolation_modes.get(interpolation)

    out_dtype = img.dtype
    need_cast = False
    if img.dtype not in (mindspore.float32, mindspore.float64):
        need_cast = True
        img = img.astype(mindspore.float32)

    img = interpolate(img, size=[size_h, size_w], mode=mode)

    if need_squeeze:
        img = mindspore.ops.Squeeze(axis=0)(img)

    if need_cast:
        img = img.astype(out_dtype)

    return img


class ColorJitter(mindspore.dataset.vision.c_transforms.RandomColorAdjust):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)


class RandomRotation(mindspore.dataset.vision.c_transforms.RandomRotation):
    def __init__(self, degrees, interpolation=PIL.Image.NEAREST, expand=False, center=None, fill=0, resample=None):
        resample = _pil_interpolation_map.get(interpolation, mindspore.dataset.vision.Inter.NEAREST)
        super().__init__(degrees, resample=resample, expand=expand, center=center, fill_value=fill)


class RandomErasing(mindspore.dataset.vision.py_transforms.RandomErasing):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__(prob=p, scale=scale, ratio=ratio, value=value, inplace=inplace)


class TenCrop(mindspore.dataset.vision.py_transforms.TenCrop):
    def __init__(self, size, vertical_flip=False):
        super().__init__(size=size, use_vertical_flip=vertical_flip)


class RandomAffine(mindspore.dataset.vision.c_transforms.RandomAffine):
    def __init__(self, degrees, translate=None, scale=None, shear=None, interpolation=PIL.Image.NEAREST, fill=0,
                 fillcolor=None, resample=None):
        _resample = _pil_interpolation_map.get(interpolation if resample is None else resample,
                                               mindspore.dataset.vision.Inter.NEAREST)
        super().__init__(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=_resample,
                         fill_value=fill if fillcolor is None else fillcolor)


class RandomApply(mindspore.dataset.transforms.c_transforms.RandomApply):
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms=transforms, prob=p)


class RandomPosterize(mindspore.dataset.vision.c_transforms.RandomPosterize):
    def __init__(self, bits, p=None):
        if p is not None:
            raise ValueError("MindSpore does not support parameter p.")
        super().__init__(bits=bits)


class RandomGrayscale(mindspore.dataset.vision.py_transforms.RandomGrayscale):
    def __init__(self, p=0.1):
        super().__init__(prob=p)


class RandomPerspective(mindspore.dataset.vision.py_transforms.RandomPerspective):
    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=PIL.Image.BILINEAR, fill=0):
        if fill != 0:
            raise ValueError("MindSpore only support fill is 0.")
        _interpolation = _pil_interpolation_map.get(interpolation, mindspore.dataset.vision.Inter.BILINEAR)
        super().__init__(distortion_scale=distortion_scale, prob=p, interpolation=_interpolation)


class Pad(mindspore.dataset.vision.c_transforms.Pad):
    def __init__(self, padding, fill=0, padding_mode='constant'):
        _pad_mode_map = {
            'constant': mindspore.dataset.vision.utils.Border.CONSTANT,
            'edge': mindspore.dataset.vision.utils.Border.EDGE,
            'reflect': mindspore.dataset.vision.utils.Border.REFLECT,
            'symmetric': mindspore.dataset.vision.utils.Border.SYMMETRIC,
        }
        padding_mode = _pad_mode_map.get(padding_mode, mindspore.dataset.vision.utils.Border.CONSTANT)
        super().__init__(paddind=padding, fill_value=fill, padding_mode=padding_mode)


class LinearTransformation(mindspore.dataset.vision.py_transforms.LinearTransformation):
    def __init__(self, transformation_matrix, mean_vector):
        if isinstance(transformation_matrix, mindspore.Tensor):
            transformation_matrix = transformation_matrix.asnumpy()
        if isinstance(mean_vector, mindspore.Tensor):
            mean_vector = mean_vector.asnumpy()
        super().__init__(transformation_matrix=transformation_matrix, mean_vector=mean_vector)


class ConvertImageDtype(mindspore.dataset.vision.py_transforms.ToType):
    def __init__(self, dtype):
        super().__init__(output_type=_ms_np_type_map.get(dtype))


class RandomSolarize(mindspore.dataset.vision.c_transforms.RandomSolarize):
    def __init__(self, threshold, p=None):
        if p is not None:
            raise ValueError("MindSpore does not support parameter p.")
        super().__init__(threshold=threshold)
