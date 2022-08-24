_base_ = 'rotated_yolox_s_8x8_300e_dota_le90.py'

# 从一个给定路径里加载模型作为预训练模型，它并不会消耗训练时间。
load_from = './work_dirs/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
