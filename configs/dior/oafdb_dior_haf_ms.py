_base_ = [
    '../_base_/datasets/obb/dior_r.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'
# fp16 = dict(loss_scale='dynamic')

# model settings
model = dict(
    type='RotatedFCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',  # use P5
            num_outs=5,
            relu_before_extra_convs=True),
        dict(
            type='ScaleFusion',
            in_channels=256,
            out_channels=256,
            num_blocks=1,
            name='HFAB')
    ],
    bbox_head=dict(
        type='OAFDHead',
        num_classes=20,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        bbox_coder=dict(
            type='DistanceAnglePointBBoxCoder', angle_range=angle_version),
        hbbox_coder=dict(type='DistancePointBBoxCoder', clip_border=False),
        norm_on_bbox=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        angel_coder=dict(
            type='CSLCoder',
            angle_version=angle_version,
            omega=1,
            window='gaussian',
            radius=1),
        loss_angle=dict(
            type='SmoothFocalLoss', gamma=2.0, alpha=0.25, loss_weight=0.2),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=[(512, 512), (1024, 1024)]),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
dataset_type = 'DIORDataset'
data_root = '/root/autodl-tmp/dataset/dior/'
data = dict(
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(type=dataset_type,
                 bbox_type='obb',
                 angle_version=angle_version,
                 ann_file=data_root + 'ImageSets/Main/train.txt',
                 ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes/',
                 img_subdir=data_root + 'JPEGImages-trainval/',
                 pipeline=train_pipeline),
            dict(type=dataset_type,
                 bbox_type='obb',
                 angle_version=angle_version,
                 ann_file=data_root + 'ImageSets/Main/val.txt',
                 ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes/',
                 img_subdir=data_root + 'JPEGImages-trainval/',
                 pipeline=train_pipeline),
        ]
    ),
    val=dict(angle_version=angle_version),
    test=dict(angle_version=angle_version))

optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001 / 4, weight_decay=0.0001)
evaluation = dict(interval=4, metric='mAP', nproc=1)
checkpoint_config = dict(interval=4)
work_dir = './work_dirs/dior/oafdb_ms'
