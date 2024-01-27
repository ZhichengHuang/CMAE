from typing import Dict, Optional, Tuple
import torch
from .base import BaseModel
from cmae.registry import MODELS

@MODELS.register_module()
class CMAE(BaseModel):
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 base_momentum=0.996,
                 init_cfg=None,
                 target_cls=True,
                 **kwargs):
        super(CMAE, self).__init__(backbone=backbone['online'],init_cfg=init_cfg)
        assert neck is not None
        # self.backbone = MODELS.build(backbone['online'])

        self.target_backbone = MODELS.build(backbone['target'])

        self.pixel_decoder = MODELS.build(neck['pixel'])
        self.feature_decoder = MODELS.build(neck['feature'])

        self.projector = MODELS.build(neck['projector'])
        self.target_projector = MODELS.build(neck['projector'])

        self.target_cls = target_cls

        assert head is not None

        self.head = MODELS.build(head)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

        for param_m in self.target_backbone.parameters():
            param_m.requires_grad = False

        for param_m in self.target_projector.parameters():
            param_m.requires_grad = False

    def init_weights(self):
        super(CMAE, self).init_weights()

        for param_b, param_m in zip(self.backbone.parameters(),
                                    self.target_backbone.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

        for param_b, param_m in zip(self.projector.parameters(),
                                    self.target_projector.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        for param_b, param_m in zip(self.backbone.parameters(),
                                    self.target_backbone.parameters()):
            param_m.data = param_m.data * self.momentum + param_b.data * (
                    1. - self.momentum)

        for param_b, param_m in zip(self.projector.parameters(),
                                    self.target_projector.parameters()):
            param_m.data = param_m.data * self.momentum + param_b.data * (
                    1. - self.momentum)


    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def forward_train(self, img,img_t=None, **kwargs):
        latent_s, mask_s, ids_restore_s = self.backbone(img)
        latent_t, mask_t, ids_restore_t = self.target_backbone(img_t,use_cls=self.target_cls)

        pred_pixel = self.pixel_decoder(latent_s,ids_restore_s)
        pred_feature = self.feature_decoder(latent_s,ids_restore_s)

        proj_s=self.projector(torch.mean(pred_feature,dim=1,keepdim=True))

        if self.target_cls:
            proj_t = self.target_projector(torch.mean(latent_t[:,1:,:], dim=1, keepdim=True))
        else:
            proj_t = self.target_projector(torch.mean(latent_t,dim=1,keepdim=True))

        losses = self.head(img,pred_pixel,mask_s,proj_s,proj_t)
        return losses

    def forward_test(self, img: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward computation during testing.

        Args:
            img (torch.Tensor): Input images of shape (N, C, H, W).
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
        mask = mask.unsqueeze(-1).repeat(1, 1, self.head.patch_size**2 *
                                         3)  # (N, H*W, p*p*3)
        mask = self.head.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        return mask, pred












