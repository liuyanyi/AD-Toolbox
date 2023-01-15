_base_ = [
    'mmdet::_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/hbb/vhr_instance.py',
    '../_base_/schedules/schedule_6x.py', '../_base_/default_runtime_det.py'
]

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=10),
        mask_head=dict(num_classes=10),
    ))

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10))
