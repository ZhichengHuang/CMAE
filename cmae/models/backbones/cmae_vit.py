
import torch
from torch import nn
from .vision_transformer import VisionTransformer


from cmae.registry import MODELS
from ..utils import build_2d_sincos_position_embedding

@MODELS.register_module()
class CMAEViT(VisionTransformer):
    """
    Vision Transformer for MAE pre-training
    """
    def __init__(self,
                 arch='b',
                 img_size=224,
                 patch_size=16,
                 out_indices=-1,
                 drop_rate=0,
                 drop_path_rate=0,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 mask_ratio=0.75,
                 init_cfg=None
                 ):
        super(CMAEViT, self).__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            output_cls_token=output_cls_token,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)
        self.pos_embed.requires_grad = False
        self.mask_ratio = mask_ratio
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]


    def init_weights(self):
        super(CMAEViT, self).init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # initialize position  embedding in backbone
            pos_embed = build_2d_sincos_position_embedding(
                int(self.num_patches**.5),
                self.pos_embed.shape[-1],
                cls_token=True)
            self.pos_embed.data.copy_(pos_embed.float())

            w = self.patch_embed.projection.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            torch.nn.init.normal_(self.cls_token, std=.02)

            self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def random_masking(self, x, mask_ratio=0.75):
    #     """Generate the mask for MAE Pre-training.
    #
    #     Args:
    #         x (torch.tensor): Image with data augmentation applied.
    #         mask_ratio (float): The mask ratio of total patches.
    #             Defaults to 0.75.
    #
    #     Returns:
    #         tuple[Tensor, Tensor, Tensor]: masked image, mask and the ids
    #             to restore original image.
    #
    #         - x_masked (Tensor): masked image.
    #         - mask (Tensor): mask used to mask image.
    #         - ids_restore (Tensor): ids to restore original image.
    #     """
    #     N, L, D = x.shape  # batch, length, dim
    #     len_keep = int(L * (1 - mask_ratio))
    #
    #     noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    #
    #     # sort noise for each sample
    #     ids_shuffle = torch.argsort(
    #         noise, dim=1)  # ascend: small is keep, large is remove
    #     ids_restore = torch.argsort(ids_shuffle, dim=1)
    #
    #     # keep the first subset
    #     ids_keep = ids_shuffle[:, :len_keep]
    #     x_masked = torch.gather(
    #         x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    #
    #     # generate the binary mask: 0 is keep, 1 is remove
    #     mask = torch.ones([N, L], device=x.device)
    #     mask[:, :len_keep] = 0
    #     # unshuffle to get the binary mask
    #     mask = torch.gather(mask, dim=1, index=ids_restore)
    #
    #     return x_masked, mask, ids_restore

    def random_masking(self,x,mask_ratio=0.65):
        N, L, D = x.shape  # batch, length, dim

        if mask_ratio>0:
            seq_len = L // 4
            len_keep = int(seq_len * (1 - mask_ratio))
            noise = torch.rand(N, seq_len, device=x.device)  # noise in [0,1]

            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_keep_s = ids_shuffle[:, :len_keep]
            mask = torch.zeros((N, seq_len), device=x.device)
            mask = torch.scatter(mask, 1, ids_keep_s, 1.0)

            mask = mask.reshape(N, int(seq_len ** 0.5), int(seq_len ** 0.5)).unsqueeze(dim=2).unsqueeze(dim=4)
            mask = mask.expand(-1, -1, 2, -1, 2).flatten(1, 2).flatten(2, 3)

            mask = mask.flatten(1)
            mask_index = torch.argsort(mask, dim=-1, descending=True)
            ids_keep = mask_index[:, :4 * len_keep]

            x_masked = torch.gather(
                x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

            ids_restore = torch.argsort(mask_index, dim=1)
            mask_out = ~mask.to(torch.bool)
            mask_out = mask_out * 1.0

            return x_masked, mask_out, ids_restore
        else:
            ids_keep = torch.arange(0,L,device=x.device,dtype=torch.long)
            ids_keep = ids_keep.unsqueeze(dim=0).repeat(N,1)
            x_masked = torch.gather(x,dim=1,index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

            ids_restore = torch.argsort(ids_keep,dim=1)
            mask_out = torch.zeros([N,L],device=x.device)

            return x_masked,mask_out,ids_restore






    def forward(self,x,use_cls=True):
        B = x.shape[0]
        x = self.patch_embed(x)[0]
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        if use_cls:
            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.drop_after_pos(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

        if self.final_norm:
            x = self.norm1(x)

        return (x, mask, ids_restore)


