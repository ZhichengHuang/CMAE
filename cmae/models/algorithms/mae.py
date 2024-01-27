# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch

from cmae.registry import MODELS
from .base import BaseModel


@MODELS.register_module()
class MAE(BaseModel):
    """MAE.

    Implementation of `Masked Autoencoders Are Scalable Vision Learners
    <https://arxiv.org/abs/2111.06377>`_.
    """

    def extract_feat(self, img: torch.Tensor) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            img (torch.Tensor): Input images of shape (N, C, H, W).
        Returns:
            Tuple[torch.Tensor]: backbone outputs.
        """
        return self.backbone(img)



    def forward_train(self, img: List[torch.Tensor],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        # ids_restore: the same as that in original repo, which is used
        # to recover the original order of tokens in decoder.
        latent, mask, ids_restore = self.backbone(img)
        pred = self.neck(latent, ids_restore)
        losses = self.head(img, pred, mask)

        return losses

    def forward_test(self, img: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward computation during testing.

        Args:
            inputs (torch.Tensor): Input images of shape (N, C, H, W).
            kwargs: Any keyword arguments to be used to forward.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output of model test.
                - mask: Mask used to mask image.
                - pred: The output of neck.
        """
        latent, mask, ids_restore = self.backbone(img)
        pred = self.neck(latent, ids_restore)

        pred = self.head.unpatchify(pred)
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.head.patch_size ** 2 *
                                         3)  # (N, H*W, p*p*3)
        mask = self.head.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        return mask, pred
