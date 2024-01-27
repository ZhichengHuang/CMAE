import json
import os


from mmengine.registry import build_from_cfg

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from cmae.registry import DATASETS,TRANSFORMS

@DATASETS.register_module()
class ImageNetDataset(Dataset):
    def __init__(self,data_root, pipeline,data_ann):
        super(ImageNetDataset, self).__init__()
        self.data_root = data_root

        pipeline = [build_from_cfg(p, TRANSFORMS) for p in pipeline]
        self.pipeline = Compose(pipeline)

        self.data_infors = self.load_annotations(os.path.join(data_root, data_ann))

    def load_annotations(self, data_ann):
        data = json.load(open(data_ann))
        self.labels = []
        for item in data:
            self.labels.append(int(item['label']))
        return data

    def __len__(self):
        return len(self.data_infors)

    def __getitem__(self, idx):
        item = self.data_infors[idx].copy()
        item['data_root'] = self.data_root
        # target = int(item['label'])
        item['label']=int(item['label'])
        item = self.pipeline(item)
        return item


        # return dict(img=img, label=target)