# Copyright (c) OpenMMLab. All rights reserved.
"""MMRotate provides 18 registry nodes to support using modules across
projects.

Each node is a child of the root registry in MMEngine.
More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine.registry import Registry

# manage data-related modules
DATASETS = Registry(
    'dataset',
    parent=MMENGINE_DATASETS,
    scope='ad',
    locations=['adtoolbox.datasets'])
# TRANSFORMS = Registry(
#     'transform',
#     parent=MMENGINE_TRANSFORMS,
#     scope='ad',
#     locations=['mmdet.datasets.transforms'])

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry(
    'model',
    parent=MMENGINE_MODELS,
    scope='ad',
    locations=['adtoolbox.models'])

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util',
    parent=MMENGINE_TASK_UTILS,
    scope='ad',
    locations=['adtoolbox.models'])
