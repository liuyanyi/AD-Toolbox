from mmdet.datasets import CocoDataset

from adtoolbox.registry import DATASETS


@DATASETS.register_module()
class VHRDataset(CocoDataset):
    CLASSES = ('airplane', 'ship', 'storage_tank', 'baseball_diamond',
               'tennis_court', 'basketball_court', 'ground_track_field',
               'harbor', 'bridge', 'vehicle')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30)]
