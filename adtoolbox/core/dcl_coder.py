import math

import numpy as np
import torch
from mmdet.core import BaseBBoxCoder
from mmrotate.core.bbox.builder import ROTATED_BBOX_CODERS


@ROTATED_BBOX_CODERS.register_module()
class DCLCoder(BaseBBoxCoder):
    def __init__(self, angle_version, coding_len=7, pos_thr=0.5):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['oc', 'le90', 'le135']
        self.angle_range = 90 if angle_version == 'oc' else 180
        self.angle_offset_dict = {'oc': 0, 'le90': 90, 'le135': 45}
        self.angle_offset = self.angle_offset_dict[angle_version]
        assert isinstance(coding_len, int)
        self.coding_len = coding_len
        self.omega = 180 / np.power(2, coding_len)
        self.pos_thr = pos_thr

    def encode(self, angle_targets):
        # radius to degree
        angle_targets_deg = angle_targets * (180 / math.pi)
        # empty label
        smooth_label = torch.zeros_like(angle_targets).repeat(
            1, self.coding_len)
        angle_targets_deg = (angle_targets_deg +
                             self.angle_offset) / self.omega
        # Float to Int
        angle_targets_long = angle_targets_deg.long()
        # print(angle_targets_long[0])

        # len~1
        for i in range(self.coding_len, 0, -1):
            smooth_label[:, i - 1:i] = angle_targets_long % 2
            angle_targets_long = angle_targets_long >> 1

        # print(smooth_label[0])
        return smooth_label

    def decode(self, angle_preds):
        power_value = np.power(2, range(self.coding_len - 1, -1, -1))
        power_value = torch.tensor(power_value,
                                   device=angle_preds.device,
                                   dtype=torch.long)

        postive_ind = angle_preds.sigmoid() > self.pos_thr

        angle_preds_bin = torch.zeros_like(angle_preds)
        angle_preds_bin[postive_ind] = 1

        angle_cls_inds = (angle_preds_bin * power_value).sum(dim=-1)
        angle_pred = ((angle_cls_inds + 0.5) *
                      self.omega) % self.angle_range - self.angle_offset
        return angle_pred * (math.pi / 180)


@ROTATED_BBOX_CODERS.register_module()
class DCLCoder2(BaseBBoxCoder):
    def __init__(self, angle_version, coding_len=7, pos_thr=0.5):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['oc', 'le90', 'le135']
        self.angle_range = 90 if angle_version == 'oc' else 180
        self.angle_offset_dict = {'oc': 0, 'le90': 90, 'le135': 45}
        self.angle_offset = self.angle_offset_dict[angle_version]
        assert isinstance(coding_len, int)
        self.coding_len = coding_len * 2
        self.omega = 180 / np.power(2, coding_len)
        self.pos_thr = pos_thr

    def encode(self, angle_targets):
        # radius to degree
        angle_targets_deg = angle_targets * (180 / math.pi)
        # empty label
        smooth_label = torch.zeros_like(angle_targets).repeat(
            1, self.coding_len)
        angle_targets_deg = (angle_targets_deg +
                             self.angle_offset) / self.omega
        # Float to Int
        angle_targets_long = angle_targets_deg.long()
        # print(angle_targets_long[0])

        # len~1
        for i in range(self.coding_len, 0, -2):
            label = (angle_targets_long % 2)
            # FTFTFT
            smooth_label[:, i - 1:i] = label == 1
            smooth_label[:, i - 2:i - 1] = label == 0
            angle_targets_long = angle_targets_long >> 1

        # print(smooth_label[0])
        return smooth_label

    def decode(self, angle_preds):
        power_value = np.power(2, range(self.coding_len // 2 - 1, -1, -1))
        power_value = torch.tensor(power_value,
                                   device=angle_preds.device,
                                   dtype=torch.long)

        angle_preds = torch.stack(
            [angle_preds[..., ::2], angle_preds[..., 1::2]], dim=-1)
        _, angle_preds_bin = angle_preds.max(dim=-1)

        angle_cls_inds = (angle_preds_bin * power_value).sum(dim=-1)
        angle_pred = ((angle_cls_inds + 0.5) *
                      self.omega) % self.angle_range - self.angle_offset
        return angle_pred * (math.pi / 180)


if __name__ == '__main__':
    coder = DCLCoder2('le90', 8, 0.5)
    input = torch.tensor([[-np.pi / 8]])
    inin = coder.encode(input)
    oo = coder.decode(inin)
    print(input)
    print(inin)
    print(oo)
