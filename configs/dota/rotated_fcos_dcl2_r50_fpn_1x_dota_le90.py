_base_ = 'rotated_fcos_sep_angle_r50_fpn_1x_dota_le90.py'
angle_version = 'le90'
fp16 = dict(loss_scale='dynamic')

# model settings
model = dict(bbox_head=dict(type='CSLRFCOSHead',
                            center_sampling=True,
                            center_sample_radius=1.5,
                            norm_on_bbox=True,
                            centerness_on_reg=True,
                            separate_angle=True,
                            scale_angle=False,
                            angle_coder=dict(type='DCLCoder2',
                                             angle_version=angle_version,
                                             coding_len=8),
                            loss_cls=dict(type='FocalLoss',
                                          use_sigmoid=True,
                                          gamma=2.0,
                                          alpha=0.25,
                                          loss_weight=1.0),
                            loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
                            loss_centerness=dict(type='CrossEntropyLoss',
                                                 use_sigmoid=True,
                                                 loss_weight=1.0),
                            loss_angle=dict(type='CrossEntropyLoss',
                                            use_sigmoid=True,
                                            loss_weight=1.0)), )
evaluation = dict(interval=12, metric='mAP', nproc=1)
