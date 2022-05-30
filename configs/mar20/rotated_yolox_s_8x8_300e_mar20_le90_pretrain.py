_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

angle_version = 'le90'
img_scale = (1024, 1024)  # height, width
fp16 = dict(loss_scale='dynamic')

# model settings
model = dict(
    type='RotatedYOLOX',
    input_size=img_scale,
    random_size_range=(25, 35),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='RotatedYOLOXHead',
        num_classes=20,
        in_channels=128,
        feat_channels=128,
        seprate_angle=True,
        # seprate_angle=False,
        # loss_bbox=dict(
        #     type='GDLoss',
        #     loss_type='kld',
        #     fun='log1p',
        #     tau=1,
        #     reduction='sum',
        #     sqrt=False,
        #     loss_weight=27.5),
        angle_coder=dict(
            type='CSLCoder',
            angle_version='le90',
            omega=1,
            window='gaussian',
            radius=1),
        loss_angle=dict(
            type='SmoothFocalLoss',
            gamma=2.0,
            alpha=0.25,
            reduction='sum',
            loss_weight=1.0),
    ),
    train_cfg=dict(assigner=dict(type='RSimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms_rotated', iou_threshold=0.10)))

# dataset settings
dataset_type = 'MAR20Dataset'
data_root = '/datasets/MAR20/'

train_pipeline = [
    dict(type='RMosaic',
         angle_version=angle_version,
         img_scale=img_scale,
         bbox_clip_border=True,
         pad_val=114.0),
    dict(
        type='RRandomAffine',
        angle_version=angle_version,
        scaling_ratio_range=(0.1, 2),
        bbox_clip_border=True,
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='RMixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        bbox_clip_border=True,
        pad_val=114.0),
    dict(type='Poly2OBB', angle_version='le90'),
    dict(type='YOLOXHSVRandomAug'),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='RResize', img_scale=img_scale),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterRotatedAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        bbox_type='obb',
        ann_file=data_root + 'ImageSets/Main/train.txt',
        ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes/',
        img_subdir=data_root + 'JPEGImages/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='OBB2Poly', angle_version='le90')
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    persistent_workers=True,
    train=train_dataset,
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
        ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes',
        img_subdir=data_root + 'JPEGImages/',
        pipeline=test_pipeline))
# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.01 / 8,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

max_epochs = 300
num_last_epochs = 15
load_from = '/workspace/toolbox/work_dirs/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'  # 从一个给定路径里加载模型作为预训练模型，它并不会消耗训练时间。
resume_from = None
interval = 10

# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        skip_type_keys=('RMosaic', 'RRandomAffine', 'RMixUp'),
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 10)],
    metric='mAP')
log_config = dict(interval=50)
