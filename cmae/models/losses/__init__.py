from .cross_entropy_loss import CrossEntropyLoss, binary_cross_entropy, cross_entropy
from .label_smooth_loss import LabelSmoothLoss

from .utils import convert_to_one_hot, reduce_loss, weight_reduce_loss, weighted_loss


__all__=[
    'LabelSmoothLoss','cross_entropy', 'binary_cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
    'weight_reduce_loss', 'LabelSmoothLoss', 'weighted_loss', 'convert_to_one_hot',
]