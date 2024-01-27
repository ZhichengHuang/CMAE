# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from mmcv.transforms import BaseTransform
from mmengine.utils import is_str

from cmae.registry import TRANSFORMS


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


@TRANSFORMS.register_module()
class Transpose(BaseTransform):
    """Transpose numpy array.

    **Required Keys:**

    - ``*keys``

    **Modified Keys:**

    - ``*keys``

    Args:
        keys (List[str]): The fields to convert to tensor.
        order (List[int]): The output dimensions order.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def transform(self, results):
        """Method to transpose array."""
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, order={self.order})'


@TRANSFORMS.register_module()
class ToPIL(BaseTransform):
    """Convert the image from OpenCV format to :obj:`PIL.Image.Image`.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    """

    def transform(self, results):
        """Method to convert images to :obj:`PIL.Image.Image`."""
        results['img'] = Image.fromarray(results['img'])
        return results


@TRANSFORMS.register_module()
class ToNumpy(BaseTransform):
    """Convert object to :obj:`numpy.ndarray`.

    **Required Keys:**

    - ``*keys**``

    **Modified Keys:**

    - ``*keys**``

    Args:
        dtype (str, optional): The dtype of the converted numpy array.
            Defaults to None.
    """

    def __init__(self, keys, dtype=None):
        self.keys = keys
        self.dtype = dtype

    def transform(self, results):
        """Method to convert object to :obj:`numpy.ndarray`."""
        for key in self.keys:
            results[key] = np.array(results[key], dtype=self.dtype)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, dtype={self.dtype})'


@TRANSFORMS.register_module()
class Collect(BaseTransform):
    """Collect and only reserve the specified fields.

    **Required Keys:**

    - ``*keys``

    **Deleted Keys:**

    All keys except those in the argument ``*keys``.

    Args:
        keys (Sequence[str]): The keys of the fields to be collected.
    """

    def __init__(
        self,
        keys,
    ):
        self.keys = keys

    def transform(self, results):
        data = {}
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class ToTensor(BaseTransform):

    def __init__(self, keys):
        self.keys = keys

    def transform(self, results):
        for key in self.keys:
            if key in ['img']:
                results[key] = to_tensor(results[key].transpose(2, 0, 1).copy())
            else:
                results[key] = to_tensor(results[key])

        return results
