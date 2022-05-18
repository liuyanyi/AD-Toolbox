from mmdet.datasets import CustomDataset
from mmdet.datasets.builder import DATASETS


@DATASETS.register_module()
class MAR20Dataset(CustomDataset):
    pass
