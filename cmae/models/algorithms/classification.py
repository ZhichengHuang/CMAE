from typing import Optional, Union

from cmae.models.utils import Augments
from timm.data.mixup import Mixup

from .base import BaseModel
from cmae.registry import MODELS


@MODELS.register_module()
class Classification(BaseModel):
    """Simple image classification.

    Args:
        backbone (dict): Config dict for module of backbone.
        with_sobel (bool): Whether to apply a Sobel filter.
            Defaults to False.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 train_cfg:Optional[dict]=None,
                 init_cfg: Optional[dict] = None):
        super(Classification, self).__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            init_cfg=init_cfg)


        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            self.augments = Augments(augments_cfg)

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img, label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        label = label.squeeze(dim=1)
        if self.augments is not None:
            img, label = self.augments(img, label)
        x = self.extract_feat(img)
        outs = self.head(x)
        loss_inputs = (outs, label)
        losses = self.head.loss(*loss_inputs)
        return losses

    def forward_test(self, img,label, **kwargs):
        """Forward computation during test.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        label = label.squeeze(dim=1)
        x = self.extract_feat(img)  # tuple
        outs = self.head(x)[0]

        out_list = []
        for pred, gt in zip(outs, label):
            tmp = dict()
            tmp['pred_score'] = pred
            tmp['gt_label'] = gt
            out_list.append(tmp)

        return out_list

    def get_layer_depth(self, param_name: str):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.

        Returns:
            Tuple[int, int]: The layer-wise depth and the max depth.
        """
        if hasattr(self.backbone, 'get_layer_depth'):
            return self.backbone.get_layer_depth(param_name, 'backbone.')
        else:
            raise NotImplementedError(
                f"The backbone {type(self.backbone)} doesn't "
                'support `get_layer_depth` by now.')



@MODELS.register_module()
class ClassificationMixup(BaseModel):
    """Simple image classification.

    Args:
        backbone (dict): Config dict for module of backbone.
        with_sobel (bool): Whether to apply a Sobel filter.
            Defaults to False.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super(ClassificationMixup, self).__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            init_cfg=init_cfg
        )

        self.augments = None
        if train_cfg is not None:
            self.augments = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=1000)

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img, label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if self.augments is not None:
            img, label = self.augments(img, label)
        x = self.extract_feat(img)
        outs = self.head(x)
        loss_inputs = (outs, label)
        losses = self.head.loss(*loss_inputs)
        return losses

    def forward_test(self, img, label, **kwargs):
        """Forward computation during test.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        x = self.extract_feat(img)  # tuple
        outs = self.head(x)[0]

        return {"pred_label":outs, "gt_label":label}
