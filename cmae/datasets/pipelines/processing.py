import inspect
import math
import numbers
import random
import warnings
from enum import EnumMeta
from typing import Optional, Sequence, Tuple, Union,List,Dict
import mmcv
from mmcv.transforms.utils import cache_randomness
from mmcv.transforms import BaseTransform


import numpy as np
from numbers import Number
import torch
import re

from mmengine.registry import build_from_cfg
from PIL import Image, ImageFilter
from timm.data import create_transform
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms.transforms import InterpolationMode


from cmae.registry import TRANSFORMS


def _str_to_torch_dtype(t: str):
    """mapping str format dtype to torch.dtype."""
    import torch  # noqa: F401,F403
    return eval(f'torch.{t}')

def _interpolation_modes_from_str(t: str):
    """mapping str format to Interpolation."""
    t = t.lower()
    inverse_modes_mapping = {
        'nearest': InterpolationMode.NEAREST,
        'bilinear': InterpolationMode.BILINEAR,
        'bicubic': InterpolationMode.BICUBIC,
        'box': InterpolationMode.BOX,
        'hammimg': InterpolationMode.HAMMING,
        'lanczos': InterpolationMode.LANCZOS,
    }

# register all existing transforms in torchvision
class TorchVisonTransformWrapper:

    def __init__(self, transform, *args, **kwargs):
        if 'interpolation' in kwargs and isinstance(kwargs['interpolation'],
                                                    str):
            kwargs['interpolation'] = _interpolation_modes_from_str(
                kwargs['interpolation'])
        if 'dtype' in kwargs and isinstance(kwargs['dtype'], str):
            kwargs['dtype'] = _str_to_torch_dtype(kwargs['dtype'])
        self.t = transform(*args, **kwargs)

    def __call__(self, results):
        results['img'] = self.t(results['img'])
        return results

    def __repr__(self) -> str:
        return f'TorchVision{repr(self.t)}'


def register_vision_transforms() -> List[str]:
    """Register transforms in ``torchvision.transforms`` to the ``TRANSFORMS``
    registry.

    Returns:
        List[str]: A list of registered transforms' name.
    """
    vision_transforms = []
    for module_name in dir(torchvision.transforms):
        if not re.match('[A-Z]', module_name):
            # must startswith a capital letter
            continue
        _transform = getattr(torchvision.transforms, module_name)
        if inspect.isclass(_transform) and callable(
                _transform) and not isinstance(_transform, (EnumMeta)):
            from functools import partial
            TRANSFORMS.register_module(
                module=partial(
                    TorchVisonTransformWrapper, transform=_transform),
                name=f'torchvision/{module_name}')
            vision_transforms.append(f'torchvision/{module_name}')
    return vision_transforms


# register all the transforms in torchvision by using a transform wrapper
VISION_TRANSFORMS = register_vision_transforms()


# @TRANSFORMS.register_module(force=True)
# class ToTensor(object):
#     """Convert image or a sequence of images to tensor.
#
#     This module can not only convert a single image to tensor, but also a
#     sequence of images.
#     """
#
#     def __init__(self) -> None:
#         self.transform = _transforms.ToTensor()
#
#     def __call__(self, imgs: Union[object, Sequence[object]]) -> torch.Tensor:
#         if isinstance(imgs, Sequence):
#             imgs = list(imgs)
#             for i, img in enumerate(imgs):
#                 imgs[i] = self.transform(img)
#         else:
#             imgs = self.transform(imgs)
#         return imgs


@TRANSFORMS.register_module()
class ShiftPixel(object):
    """
    Args:

    """

    def __init__(self, pixel=31,size=224):

        self.pixel = pixel
        self.size= size

    def __call__(self, results):
        img = results['img']
        pixel_h=random.randint(0,self.pixel)
        pixel_w= random.randint(0,self.pixel)
        img = np.array(img)
        w = img.shape[0]
        assert pixel_h+self.size<w
        assert pixel_w+self.size<w

        img = img[0+pixel_h:pixel_h+224,0+pixel_w:pixel_w+224,:]
        # return Image.fromarray(img.astype(np.uint8))
        results['img']=img.astype(np.uint8)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'pixel = {self.pixel}, '
        repr_str += f'size = {self.size}'
        return repr_str

@TRANSFORMS.register_module()
class RandomResizedCropAndInterpolationWithTwoPic(object):
    """Crop the given PIL Image to random size and aspect ratio with random
    interpolation.

    This module is borrowed from
    https://github.com/microsoft/unilm/tree/master/beit.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size. This is popularly used
    to train the Inception networks. This module first crops the image and
    resizes the crop to two different sizes.

    Args:
        size (Union[tuple, int]): Expected output size of each edge of the
            first image.
        second_size (Union[tuple, int], optional): Expected output size of each
            edge of the second image.
        scale (tuple[float, float]): Range of size of the origin size cropped.
            Defaults to (0.08, 1.0).
        ratio (tuple[float, float]): Range of aspect ratio of the origin aspect
            ratio cropped. Defaults to (3./4., 4./3.).
        interpolation (str): The interpolation for the first image. Defaults
            to ``bilinear``.
        second_interpolation (str): The interpolation for the second image.
            Defaults to ``lanczos``.
    """

    interpolation_dict = {
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS,
        'hamming': Image.HAMMING
    }

    def __init__(self,
                 size: Union[tuple, int],
                 second_size=None,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear',
                 second_interpolation='lanczos') -> None:
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if second_size is not None:
            if isinstance(second_size, tuple):
                self.second_size = second_size
            else:
                self.second_size = (second_size, second_size)
        else:
            self.second_size = None
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn('range should be of kind (min, max)')

        if interpolation == 'random':
            self.interpolation = (Image.BILINEAR, Image.BICUBIC)
        else:
            self.interpolation = self.interpolation_dict.get(
                interpolation, Image.BILINEAR)
        self.second_interpolation = self.interpolation_dict.get(
            second_interpolation, Image.BILINEAR)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: np.ndarray, scale: tuple,
                   ratio: tuple) -> Sequence[int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (np.ndarray): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect
                ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(
            self, img: np.ndarray
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if self.second_size is None:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation)
        else:
            return F.resized_crop(img, i, j, h, w, self.size,
                                  interpolation), F.resized_crop(
                                      img, i, j, h, w, self.second_size,
                                      self.second_interpolation)


# @TRANSFORMS.register_module()
# class RandomAug(object):
#     """RandAugment data augmentation method based on
#     `"RandAugment: Practical automated data augmentation
#     with a reduced search space"
#     <https://arxiv.org/abs/1909.13719>`_.
#
#     This code is borrowed from <https://github.com/pengzhiliang/MAE-pytorch>
#     """
#
#     def __init__(self,
#                  input_size=None,
#                  color_jitter=None,
#                  auto_augment=None,
#                  interpolation=None,
#                  re_prob=None,
#                  re_mode=None,
#                  re_count=None,
#                  mean=None,
#                  std=None):
#
#         self.trans = create_transform(
#             input_size=input_size,
#             is_training=True,
#             color_jitter=color_jitter,
#             auto_augment=auto_augment,
#             interpolation=interpolation,
#             re_prob=re_prob,
#             re_mode=re_mode,
#             re_count=re_count,
#             mean=mean,
#             std=std,
#         )
#
#     def __call__(self, img):
#         return self.trans(img)
#
#     def __repr__(self) -> str:
#         repr_str = self.__class__.__name__
#         return repr_str


# @TRANSFORMS.register_module()
# class RandomAppliedTrans(object):
#     """Randomly applied transformations.
#
#     Args:
#         transforms (list[dict]): List of transformations in dictionaries.
#         p (float, optional): Probability. Defaults to 0.5.
#     """
#
#     def __init__(self, transforms, p=0.5):
#         t = [build_from_cfg(t, TRANSFORMS) for t in transforms]
#         self.trans = TRANSFORMS.RandomApply(t, p=p)
#         self.prob = p
#
#     def __call__(self, img):
#         return self.trans(img)
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'prob = {self.prob}'
#         return repr_str


# custom transforms
# @TRANSFORMS.register_module()
# class Lighting(object):
#     """Lighting noise(AlexNet - style PCA - based noise).
#
#     Args:
#         alphastd (float, optional): The parameter for Lighting.
#             Defaults to 0.1.
#     """
#
#     _IMAGENET_PCA = {
#         'eigval':
#         torch.Tensor([0.2175, 0.0188, 0.0045]),
#         'eigvec':
#         torch.Tensor([
#             [-0.5675, 0.7192, 0.4009],
#             [-0.5808, -0.0045, -0.8140],
#             [-0.5836, -0.6948, 0.4203],
#         ])
#     }
#
#     def __init__(self, alphastd=0.1):
#         self.alphastd = alphastd
#         self.eigval = self._IMAGENET_PCA['eigval']
#         self.eigvec = self._IMAGENET_PCA['eigvec']
#
#     def __call__(self, img):
#         assert isinstance(img, torch.Tensor), \
#             f'Expect torch.Tensor, got {type(img)}'
#         if self.alphastd == 0:
#             return img
#
#         alpha = img.new().resize_(3).normal_(0, self.alphastd)
#         rgb = self.eigvec.type_as(img).clone()\
#             .mul(alpha.view(1, 3).expand(3, 3))\
#             .mul(self.eigval.view(1, 3).expand(3, 3))\
#             .sum(1).squeeze()
#
#         return img.add(rgb.view(3, 1, 1).expand_as(img))
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'alphastd = {self.alphastd}'
#         return repr_str


# @TRANSFORMS.register_module()
# class RandomGaussianBlur(object):
#     """RandomGaussianBlur augmentation refers to `SimCLR.
#
#     <https://arxiv.org/abs/2002.05709>`_.
#
#     Args:
#         sigma_min (float): The minimum parameter of Gaussian kernel std.
#         sigma_max (float): The maximum parameter of Gaussian kernel std.
#         p (float, optional): Probability. Defaults to 0.5.
#     """
#
#     def __init__(self, sigma_min, sigma_max, prob=0.5):
#         assert 0 <= prob <= 1.0, \
#             f'The prob should be in range [0,1], got {prob} instead.'
#         self.sigma_min = sigma_min
#         self.sigma_max = sigma_max
#         self.prob = prob
#
#     def __call__(self, results):
#         if np.random.rand() > self.prob:
#             return results
#         img = results['img']
#         sigma = np.random.uniform(self.sigma_min, self.sigma_max)
#         img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
#         results['img']=img
#         return results
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'sigma_min = {self.sigma_min}, '
#         repr_str += f'sigma_max = {self.sigma_max}, '
#         repr_str += f'prob = {self.prob}'
#         return repr_str


# @TRANSFORMS.register_module()
# class Solarization(object):
#     """Solarization augmentation refers to `BYOL.
#
#     <https://arxiv.org/abs/2006.07733>`_.
#
#     Args:
#         threshold (float, optional): The solarization threshold.
#             Defaults to 128.
#         p (float, optional): Probability. Defaults to 0.5.
#     """
#
#     def __init__(self, threshold=128, p=0.5):
#         assert 0 <= p <= 1.0, \
#             f'The prob should be in range [0, 1], got {p} instead.'
#
#         self.threshold = threshold
#         self.prob = p
#
#     def __call__(self, img):
#         if np.random.rand() > self.prob:
#             return img
#         img = np.array(img)
#         img = np.where(img < self.threshold, img, 255 - img)
#         return Image.fromarray(img.astype(np.uint8))
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'threshold = {self.threshold}, '
#         repr_str += f'prob = {self.prob}'
#         return repr_str
#

@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """Crop the given Image at a random location.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    Args:
        crop_size (int | Sequence): Desired output size of the crop. If
            crop_size is an int instead of sequence like (h, w), a square crop
            (crop_size, crop_size) is made.
        padding (int | Sequence, optional): Optional padding on each border
            of the image. If a sequence of length 4 is provided, it is used to
            pad left, top, right, bottom borders respectively.  If a sequence
            of length 2 is provided, it is used to pad left/right, top/bottom
            borders, respectively. Default: None, which means no padding.
        pad_if_needed (bool): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
            Default: False.
        pad_val (Number | Sequence[Number]): Pixel pad_val value for constant
            fill. If a tuple of length 3, it is used to pad_val R, G, B
            channels respectively. Default: 0.
        padding_mode (str): Type of padding. Defaults to "constant". Should
            be one of the following:

            - ``constant``: Pads with a constant value, this value is specified
              with pad_val.
            - ``edge``: pads with the last value at the edge of the image.
            - ``reflect``: Pads with reflection of image without repeating the
              last value on the edge. For example, padding [1, 2, 3, 4]
              with 2 elements on both sides in reflect mode will result
              in [3, 2, 1, 2, 3, 4, 3, 2].
            - ``symmetric``: Pads with reflection of image repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with
              2 elements on both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3].
    """

    def __init__(self,
                 crop_size: Union[Sequence, int],
                 padding: Optional[Union[Sequence, int]] = None,
                 pad_if_needed: bool = False,
                 pad_val: Union[Number, Sequence[Number]] = 0,
                 padding_mode: str = 'constant'):
        if isinstance(crop_size, Sequence):
            assert len(crop_size) == 2
            assert crop_size[0] > 0 and crop_size[1] > 0
            self.crop_size = crop_size
        else:
            assert crop_size > 0
            self.crop_size = (crop_size, crop_size)
        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    @cache_randomness
    def rand_crop_params(self, img: np.ndarray):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.

        Returns:
            tuple: Params (offset_h, offset_w, target_h, target_w) to be
                passed to ``crop`` for random crop.
        """
        h, w = img.shape[:2]
        target_h, target_w = self.crop_size
        if w == target_w and h == target_h:
            return 0, 0, h, w
        elif w < target_w or h < target_h:
            target_w = min(w, target_w)
            target_h = min(h, target_h)

        offset_h = np.random.randint(0, h - target_h + 1)
        offset_w = np.random.randint(0, w - target_w + 1)

        return offset_h, offset_w, target_h, target_w

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        img = results['img']
        if self.padding is not None:
            img = mmcv.impad(img, padding=self.padding, pad_val=self.pad_val)

        # pad img if needed
        if self.pad_if_needed:
            h_pad = math.ceil(max(0, self.crop_size[0] - img.shape[0]) / 2)
            w_pad = math.ceil(max(0, self.crop_size[1] - img.shape[1]) / 2)

            img = mmcv.impad(
                img,
                padding=(w_pad, h_pad, w_pad, h_pad),
                pad_val=self.pad_val,
                padding_mode=self.padding_mode)

        offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
        img = mmcv.imcrop(
            img,
            np.array([
                offset_w,
                offset_h,
                offset_w + target_w - 1,
                offset_h + target_h - 1,
            ]))
        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(crop_size={self.crop_size}'
        repr_str += f', padding={self.padding}'
        repr_str += f', pad_if_needed={self.pad_if_needed}'
        repr_str += f', pad_val={self.pad_val}'
        repr_str += f', padding_mode={self.padding_mode})'
        return repr_str


@TRANSFORMS.register_module()
class RandomResizedCrop(BaseTransform):
    """Crop the given image to random scale and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    Args:
        scale (sequence | int): Desired output scale of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        crop_ratio_range (tuple): Range of the random size of the cropped
            image compared to the original image. Defaults to (0.08, 1.0).
        aspect_ratio_range (tuple): Range of the random aspect ratio of the
            cropped image compared to the original image.
            Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maximum number of attempts before falling back to
            Central Crop. Defaults to 10.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bilinear'.
        backend (str): The image resize backend type, accepted values are
            'cv2' and 'pillow'. Defaults to 'cv2'.
    """

    def __init__(self,
                 scale: Union[Sequence, int],
                 crop_ratio_range: Tuple[float, float] = (0.08, 1.0),
                 aspect_ratio_range: Tuple[float, float] = (3. / 4., 4. / 3.),
                 max_attempts: int = 10,
                 interpolation: str = 'bilinear',
                 backend: str = 'cv2') -> None:
        if isinstance(scale, Sequence):
            assert len(scale) == 2
            assert scale[0] > 0 and scale[1] > 0
            self.scale = scale
        else:
            assert scale > 0
            self.scale = (scale, scale)
        if (crop_ratio_range[0] > crop_ratio_range[1]) or (
                aspect_ratio_range[0] > aspect_ratio_range[1]):
            raise ValueError(
                'range should be of kind (min, max). '
                f'But received crop_ratio_range {crop_ratio_range} '
                f'and aspect_ratio_range {aspect_ratio_range}.')
        assert isinstance(max_attempts, int) and max_attempts >= 0, \
            'max_attempts mush be int and no less than 0.'
        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')

        self.crop_ratio_range = crop_ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.max_attempts = max_attempts
        self.interpolation = interpolation
        self.backend = backend

    @cache_randomness
    def rand_crop_params(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.

        Returns:
            tuple: Params (offset_h, offset_w, target_h, target_w) to be
                passed to `crop` for a random sized crop.
        """
        h, w = img.shape[:2]
        area = h * w

        for _ in range(self.max_attempts):
            target_area = np.random.uniform(*self.crop_ratio_range) * area
            log_ratio = (math.log(self.aspect_ratio_range[0]),
                         math.log(self.aspect_ratio_range[1]))
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))
            target_w = int(round(math.sqrt(target_area * aspect_ratio)))
            target_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_w <= w and 0 < target_h <= h:
                offset_h = np.random.randint(0, h - target_h + 1)
                offset_w = np.random.randint(0, w - target_w + 1)

                return offset_h, offset_w, target_h, target_w

        # Fallback to central crop
        in_ratio = float(w) / float(h)
        if in_ratio < min(self.aspect_ratio_range):
            target_w = w
            target_h = int(round(target_w / min(self.aspect_ratio_range)))
        elif in_ratio > max(self.aspect_ratio_range):
            target_h = h
            target_w = int(round(target_h * max(self.aspect_ratio_range)))
        else:  # whole image
            target_w = w
            target_h = h
        offset_h = (h - target_h) // 2
        offset_w = (w - target_w) // 2
        return offset_h, offset_w, target_h, target_w

    def transform(self, results: dict) -> dict:
        """Transform function to randomly resized crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly resized cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        img = results['img']
        offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
        img = mmcv.imcrop(
            img,
            bboxes=np.array([
                offset_w, offset_h, offset_w + target_w - 1,
                offset_h + target_h - 1
            ]))
        img = mmcv.imresize(
            img,
            tuple(self.scale[::-1]),
            interpolation=self.interpolation,
            backend=self.backend)
        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(scale={self.scale}'
        repr_str += ', crop_ratio_range='
        repr_str += f'{tuple(round(s, 4) for s in self.crop_ratio_range)}'
        repr_str += ', aspect_ratio_range='
        repr_str += f'{tuple(round(r, 4) for r in self.aspect_ratio_range)}'
        repr_str += f', max_attempts={self.max_attempts}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str


@TRANSFORMS.register_module()
class RandomErasing(BaseTransform):
    """Randomly selects a rectangle region in an image and erase pixels.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img

    Args:
        erase_prob (float): Probability that image will be randomly erased.
            Default: 0.5
        min_area_ratio (float): Minimum erased area / input image area
            Default: 0.02
        max_area_ratio (float): Maximum erased area / input image area
            Default: 0.4
        aspect_range (sequence | float): Aspect ratio range of erased area.
            if float, it will be converted to (aspect_ratio, 1/aspect_ratio)
            Default: (3/10, 10/3)
        mode (str): Fill method in erased area, can be:

            - const (default): All pixels are assign with the same value.
            - rand: each pixel is assigned with a random value in [0, 255]

        fill_color (sequence | Number): Base color filled in erased area.
            Defaults to (128, 128, 128).
        fill_std (sequence | Number, optional): If set and ``mode`` is 'rand',
            fill erased area with random color from normal distribution
            (mean=fill_color, std=fill_std); If not set, fill erased area with
            random color from uniform distribution (0~255). Defaults to None.

    Note:
        See `Random Erasing Data Augmentation
        <https://arxiv.org/pdf/1708.04896.pdf>`_

        This paper provided 4 modes: RE-R, RE-M, RE-0, RE-255, and use RE-M as
        default. The config of these 4 modes are:

        - RE-R: RandomErasing(mode='rand')
        - RE-M: RandomErasing(mode='const', fill_color=(123.67, 116.3, 103.5))
        - RE-0: RandomErasing(mode='const', fill_color=0)
        - RE-255: RandomErasing(mode='const', fill_color=255)
    """

    def __init__(self,
                 erase_prob=0.5,
                 min_area_ratio=0.02,
                 max_area_ratio=0.4,
                 aspect_range=(3 / 10, 10 / 3),
                 mode='const',
                 fill_color=(128, 128, 128),
                 fill_std=None):
        assert isinstance(erase_prob, float) and 0. <= erase_prob <= 1.
        assert isinstance(min_area_ratio, float) and 0. <= min_area_ratio <= 1.
        assert isinstance(max_area_ratio, float) and 0. <= max_area_ratio <= 1.
        assert min_area_ratio <= max_area_ratio, \
            'min_area_ratio should be smaller than max_area_ratio'
        if isinstance(aspect_range, float):
            aspect_range = min(aspect_range, 1 / aspect_range)
            aspect_range = (aspect_range, 1 / aspect_range)
        assert isinstance(aspect_range, Sequence) and len(aspect_range) == 2 \
            and all(isinstance(x, float) for x in aspect_range), \
            'aspect_range should be a float or Sequence with two float.'
        assert all(x > 0 for x in aspect_range), \
            'aspect_range should be positive.'
        assert aspect_range[0] <= aspect_range[1], \
            'In aspect_range (min, max), min should be smaller than max.'
        assert mode in ['const', 'rand'], \
            'Please select `mode` from ["const", "rand"].'
        if isinstance(fill_color, Number):
            fill_color = [fill_color] * 3
        assert isinstance(fill_color, Sequence) and len(fill_color) == 3 \
            and all(isinstance(x, Number) for x in fill_color), \
            'fill_color should be a float or Sequence with three int.'
        if fill_std is not None:
            if isinstance(fill_std, Number):
                fill_std = [fill_std] * 3
            assert isinstance(fill_std, Sequence) and len(fill_std) == 3 \
                and all(isinstance(x, Number) for x in fill_std), \
                'fill_std should be a float or Sequence with three int.'

        self.erase_prob = erase_prob
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_range = aspect_range
        self.mode = mode
        self.fill_color = fill_color
        self.fill_std = fill_std

    def _fill_pixels(self, img, top, left, h, w):
        """Fill pixels to the patch of image."""
        if self.mode == 'const':
            patch = np.empty((h, w, 3), dtype=np.uint8)
            patch[:, :] = np.array(self.fill_color, dtype=np.uint8)
        elif self.fill_std is None:
            # Uniform distribution
            patch = np.random.uniform(0, 256, (h, w, 3)).astype(np.uint8)
        else:
            # Normal distribution
            patch = np.random.normal(self.fill_color, self.fill_std, (h, w, 3))
            patch = np.clip(patch.astype(np.int32), 0, 255).astype(np.uint8)

        img[top:top + h, left:left + w] = patch
        return img

    @cache_randomness
    def random_disable(self):
        """Randomly disable the transform."""
        return np.random.rand() > self.erase_prob

    @cache_randomness
    def random_patch(self, img_h, img_w):
        """Randomly generate patch the erase."""
        # convert the aspect ratio to log space to equally handle width and
        # height.
        log_aspect_range = np.log(
            np.array(self.aspect_range, dtype=np.float32))
        aspect_ratio = np.exp(np.random.uniform(*log_aspect_range))
        area = img_h * img_w
        area *= np.random.uniform(self.min_area_ratio, self.max_area_ratio)

        h = min(int(round(np.sqrt(area * aspect_ratio))), img_h)
        w = min(int(round(np.sqrt(area / aspect_ratio))), img_w)
        top = np.random.randint(0, img_h - h) if img_h > h else 0
        left = np.random.randint(0, img_w - w) if img_w > w else 0
        return top, left, h, w

    def transform(self, results):
        """
        Args:
            results (dict): Results dict from pipeline

        Returns:
            dict: Results after the transformation.
        """
        if self.random_disable():
            return results

        img = results['img']
        img_h, img_w = img.shape[:2]

        img = self._fill_pixels(img, *self.random_patch(img_h, img_w))

        results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(erase_prob={self.erase_prob}, '
        repr_str += f'min_area_ratio={self.min_area_ratio}, '
        repr_str += f'max_area_ratio={self.max_area_ratio}, '
        repr_str += f'aspect_range={self.aspect_range}, '
        repr_str += f'mode={self.mode}, '
        repr_str += f'fill_color={self.fill_color}, '
        repr_str += f'fill_std={self.fill_std})'
        return repr_str


@TRANSFORMS.register_module()
class ResizeEdge(BaseTransform):
    """Resize images along the specified edge.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    **Added Keys:**

    - scale
    - scale_factor

    Args:
        scale (int): The edge scale to resizing.
        edge (str): The edge to resize. Defaults to 'short'.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results.
            Defaults to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
            Defaults to 'bilinear'.
    """

    def __init__(self,
                 scale: int,
                 edge: str = 'short',
                 backend: str = 'cv2',
                 interpolation: str = 'bilinear') -> None:
        allow_edges = ['short', 'long', 'width', 'height']
        assert edge in allow_edges, \
            f'Invalid edge "{edge}", please specify from {allow_edges}.'
        self.edge = edge
        self.scale = scale
        self.backend = backend
        self.interpolation = interpolation

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        img, w_scale, h_scale = mmcv.imresize(
            results['img'],
            results['scale'],
            interpolation=self.interpolation,
            return_scale=True,
            backend=self.backend)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['scale'] = img.shape[:2][::-1]
        results['scale_factor'] = (w_scale, h_scale)

    def transform(self, results: Dict) -> Dict:
        """Transform function to resize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img', 'scale', 'scale_factor',
            'img_shape' keys are updated in result dict.
        """
        assert 'img' in results, 'No `img` field in the input.'

        h, w = results['img'].shape[:2]
        if any([
                # conditions to resize the width
                self.edge == 'short' and w < h,
                self.edge == 'long' and w > h,
                self.edge == 'width',
        ]):
            width = self.scale
            height = int(self.scale * h / w)
        else:
            height = self.scale
            width = int(self.scale * w / h)
        results['scale'] = (width, height)

        self._resize_img(results)
        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'edge={self.edge}, '
        repr_str += f'backend={self.backend}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class ColorJitter(BaseTransform):
    """Randomly change the brightness, contrast and saturation of an image.

    Modified from
    https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
    Licensed under the BSD 3-Clause License.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img

    Args:
        brightness (float | Sequence[float] (min, max)): How much to jitter
            brightness. brightness_factor is chosen uniformly from
            ``[max(0, 1 - brightness), 1 + brightness]`` or the given
            ``[min, max]``. Should be non negative numbers. Defaults to 0.
        contrast (float | Sequence[float] (min, max)): How much to jitter
            contrast. contrast_factor is chosen uniformly from
            ``[max(0, 1 - contrast), 1 + contrast]`` or the given
            ``[min, max]``. Should be non negative numbers. Defaults to 0.
        saturation (float | Sequence[float] (min, max)): How much to jitter
            saturation. saturation_factor is chosen uniformly from
            ``[max(0, 1 - saturation), 1 + saturation]`` or the given
            ``[min, max]``. Should be non negative numbers. Defaults to 0.
        hue (float | Sequence[float] (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from ``[-hue, hue]`` (0 <= hue
            <= 0.5) or the given ``[min, max]`` (-0.5 <= min <= max <= 0.5).
            Defaults to 0.
        backend (str): The backend to operate the image. Defaults to 'pillow'
    """

    def __init__(self,
                 brightness: Union[float, Sequence[float]] = 0.,
                 contrast: Union[float, Sequence[float]] = 0.,
                 saturation: Union[float, Sequence[float]] = 0.,
                 hue: Union[float, Sequence[float]] = 0.,
                 backend='pillow'):
        self.brightness = self._set_range(brightness, 'brightness')
        self.contrast = self._set_range(contrast, 'contrast')
        self.saturation = self._set_range(saturation, 'saturation')
        self.hue = self._set_range(hue, 'hue', center=0, bound=(-0.5, 0.5))
        self.backend = backend

    def _set_range(self, value, name, center=1, bound=(0, float('inf'))):
        """Set the range of magnitudes."""
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    f'If {name} is a single number, it must be non negative.')
            value = (center - float(value), center + float(value))

        if isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                value = np.clip(value, bound[0], bound[1])
                from mmengine.logging import MMLogger
                logger = MMLogger.get_current_instance()
                logger.warning(f'ColorJitter {name} values exceed the bound '
                               f'{bound}, clipped to the bound.')
        else:
            raise TypeError(f'{name} should be a single number '
                            'or a list/tuple with length 2.')

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        else:
            value = tuple(value)

        return value

    @cache_randomness
    def _rand_params(self):
        """Get random parameters including magnitudes and indices of
        transforms."""
        trans_inds = np.random.permutation(4)
        b, c, s, h = (None, ) * 4

        if self.brightness is not None:
            b = np.random.uniform(self.brightness[0], self.brightness[1])
        if self.contrast is not None:
            c = np.random.uniform(self.contrast[0], self.contrast[1])
        if self.saturation is not None:
            s = np.random.uniform(self.saturation[0], self.saturation[1])
        if self.hue is not None:
            h = np.random.uniform(self.hue[0], self.hue[1])

        return trans_inds, b, c, s, h

    def transform(self, results: Dict) -> Dict:
        """Transform function to resize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: ColorJitter results, 'img' key is updated in result dict.
        """
        img = results['img']
        trans_inds, brightness, contrast, saturation, hue = self._rand_params()

        for index in trans_inds:
            if index == 0 and brightness is not None:
                img = mmcv.adjust_brightness(
                    img, brightness, backend=self.backend)
            elif index == 1 and contrast is not None:
                img = mmcv.adjust_contrast(img, contrast, backend=self.backend)
            elif index == 2 and saturation is not None:
                img = mmcv.adjust_color(
                    img, alpha=saturation, backend=self.backend)
            elif index == 3 and hue is not None:
                img = mmcv.adjust_hue(img, hue, backend=self.backend)

        results['img'] = img
        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(brightness={self.brightness}, '
        repr_str += f'contrast={self.contrast}, '
        repr_str += f'saturation={self.saturation}, '
        repr_str += f'hue={self.hue})'
        return repr_str
