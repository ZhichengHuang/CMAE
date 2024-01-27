import os
import json

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from mmengine.registry import build_from_cfg

from cmae.registry import DATASETS,TRANSFORMS


@DATASETS.register_module()
class CMAEDataset(Dataset):
    def __init__(self,data_root, data_ann, pipeline,pixel=31, test=False):
        self.data_root = data_root

        self.data_infors = self.load_annotations(os.path.join(data_root, data_ann))
        self.test = test

        pipeline_base = [build_from_cfg(p, TRANSFORMS) for p in pipeline[:3]]
        self.pipeline_base = Compose(pipeline_base)
        pipeline_final = [build_from_cfg(p, TRANSFORMS) for p in pipeline[3:]]
        self.shift = build_from_cfg(dict(
            type='ShiftPixel',
            pixel=0), TRANSFORMS)
        self.pipeline_final = Compose(pipeline_final)

        pipeline_aug = [
            dict(
                type='ShiftPixel',
                pixel=pixel,
            ),
            dict(
                type='RandomApply',
                transforms=[
                    dict(
                        type='ColorJitter',
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1)
                ],
                prob=0.8),
            dict(type='RandomGrayscale', prob=0.2,keep_channels=True, channel_weights=(0.114, 0.587, 0.2989)),
            dict(type='GaussianBlur', magnitude_range=(0.1, 2.0), magnitude_std='inf', prob=0.5)
        ]
        pipeline_aug_l = [build_from_cfg(p, TRANSFORMS) for p in pipeline_aug]
        self.pipeline_aug = Compose(pipeline_aug_l)



    def load_annotations(self,data_ann):
        data = json.load(open(data_ann))
        return data

    def __len__(self):
        return len(self.data_infors)

    def __getitem__(self, idx):
        item = self.data_infors[idx].copy()
        item['data_root'] = self.data_root
        src_img = self.pipeline_base(item)

        patch_results = {'img': src_img['img']}
        img_t_results = {'img': src_img['img'].copy()}

        patch = self.pipeline_final(self.shift(patch_results))

        img_t = self.pipeline_final(self.pipeline_aug(img_t_results))

        out = {'img':patch['img'],'img_t':img_t['img']}

        return out

