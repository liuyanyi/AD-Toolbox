_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/hbb/mar20.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]
# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)), )
