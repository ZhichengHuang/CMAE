import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from mmengine.registry import build_from_cfg

from cmae.registry import DATASETS,TRANSFORMS


@DATASETS.register_module()
class MAEDataset(Dataset):
    def __init__(self,data_root, data_ann, pipeline, test=False):
        self.data_root = data_root

        self.data_infors = self.load_annotations(os.path.join(data_root, data_ann))
        self.test = test

        pipeline1 = [build_from_cfg(p, TRANSFORMS) for p in pipeline]
        self.pipeline = Compose(pipeline1)

    def load_annotations(self,data_ann):
        data = json.load(open(data_ann))
        return data

    def __len__(self):
        return len(self.data_infors)

    def __getitem__(self, idx):
        item = self.data_infors[idx].copy()
        item['data_root'] = self.data_root
        patch = self.pipeline(item)
        return patch

