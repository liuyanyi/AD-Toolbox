# dataset settings
dataset_type = 'MAR20Dataset'
data_root = '/datasets/MAR20/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        bbox_type='obb',
        ann_file=data_root + 'ImageSets/Main/train.txt',
        ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes/',
        img_subdir=data_root + 'JPEGImages/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        bbox_type='obb',
        ann_file=data_root + 'ImageSets/Main/test.txt',
        ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes/',
        img_subdir=data_root + 'JPEGImages/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        bbox_type='obb',
        ann_file=data_root + 'ImageSets/Main/test.txt',
        ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes/',
        img_subdir=data_root + 'JPEGImages/',
        pipeline=test_pipeline))
evaluation = dict(metric='mAP', interval=2)
