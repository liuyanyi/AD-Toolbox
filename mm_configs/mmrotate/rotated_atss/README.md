# Rotated ATSS

> [Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection](https://arxiv.org/abs/1912.02424)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/atss.jpg" width="800"/>
</div>

Object detection has been dominated by anchor-based detectors for several years. Recently, anchor-free detectors have
become popular due to the proposal of FPN and Focal Loss. In this paper, we first point out that the essential
difference between anchor-based and anchor-free detection is actually how to define positive and negative training
samples, which leads to the performance gap between them. If they adopt the same definition of positive and negative
samples during training, there is no obvious difference in the final performance, no matter regressing from a box or a
point. This shows that how to select positive and negative training samples is important for current object detectors.
Then, we propose an Adaptive Training Sample Selection (ATSS) to automatically select positive and negative samples
according to statistical characteristics of object. It significantly improves the performance of anchor-based and
anchor-free detectors and bridges the gap between them. Finally, we discuss the necessity of tiling multiple anchors per
location on the image to detect objects. Extensive experiments conducted on MS COCO support our aforementioned analysis
and conclusions. With the newly introduced ATSS, we improve state-of-the-art detectors by a large margin to 50.7% AP
without introducing any overhead.

## Results and Models

DOTA1.0

| Backbone | mAP | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size | Configs | Download |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :
----------------------------------------------------------------------------------------------------------------: | :
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:
|
| ResNet50 (1024,1024,200) | 64.55 | oc | 1x | 3.38 | 15.7 | - | 2
|    [rotated_retinanet_hbb_r50_fpn_1x_dota_oc](../rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc.py)
|       [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_r50_fpn_1x_dota_oc-e8a7c7df.pth)
\| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_r50_fpn_1x_dota_oc_20220121_095315.log.json)
|
| ResNet50 (1024,1024,200) | 68.42 | le90 | 1x | 3.38 | 16.9 | - | 2
|  [rotated_retinanet_obb_r50_fpn_1x_dota_le90](../rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py)
|   [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth)
\| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90/rotated_retinanet_obb_r50_fpn_1x_dota_le90_20220128_130740.log.json)
|
| ResNet50 (1024,1024,200) | 69.79 | le135 | 1x | 3.38 | 17.2 | - | 2
| [rotated_retinanet_obb_r50_fpn_1x_dota_le135](../rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135.py)
| [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135/rotated_retinanet_obb_r50_fpn_1x_dota_le135-e4131166.pth)
\| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135/rotated_retinanet_obb_r50_fpn_1x_dota_le135_20220128_130755.log.json)
|
| | | | | | | | | | |
| ResNet50 (1024,1024,200) | 65.59 | oc | 1x | 3.12 | 18.5 | - | 2
|                  [rotated_atss_hbb_r50_fpn_1x_dota_oc](./rotated_atss_hbb_r50_fpn_1x_dota_oc.py)
|                      [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_atss/rotated_atss_hbb_r50_fpn_1x_dota_oc/rotated_atss_hbb_r50_fpn_1x_dota_oc-eaa94033.pth)
\| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_atss/rotated_atss_hbb_r50_fpn_1x_dota_oc/rotated_atss_hbb_r50_fpn_1x_dota_oc_20220402_002230.log.json)
|
| ResNet50 (1024,1024,200) | 70.64 | le90 | 1x | 3.12 | 18.2 | - | 2
|                [rotated_atss_obb_r50_fpn_1x_dota_le90](./rotated_atss_obb_r50_fpn_1x_dota_le90.py)
|                  [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_atss/rotated_atss_obb_r50_fpn_1x_dota_le90/rotated_atss_obb_r50_fpn_1x_dota_le90-e029ca06.pth)
\| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_atss/rotated_atss_obb_r50_fpn_1x_dota_le90/rotated_atss_obb_r50_fpn_1x_dota_le90_20220402_002048.log.json)
|
| ResNet50 (1024,1024,200) | 72.29 | le135 | 1x | 3.19 | 18.8 | - | 2
|               [rotated_atss_obb_r50_fpn_1x_dota_le135](./rotated_atss_obb_r50_fpn_1x_dota_le135.py)
|                [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_atss/rotated_atss_obb_r50_fpn_1x_dota_le135/rotated_atss_obb_r50_fpn_1x_dota_le135-eab7bc12.pth)
\| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_atss/rotated_atss_obb_r50_fpn_1x_dota_le135/rotated_atss_obb_r50_fpn_1x_dota_le135_20220402_002138.log.json)
|
| | | | | | | | | | |
| ResNet50 (1024,1024,200) | 69.80 | oc | 1x | 3.54 | 12.4 | - | 2
|                          [r3det_r50_fpn_1x_dota_oc](../r3det/r3det_r50_fpn_1x_dota_oc.py)
|                                                   [model](https://download.openmmlab.com/mmrotate/v0.1.0/r3det/r3det_r50_fpn_1x_dota_oc/r3det_r50_fpn_1x_dota_oc-b1fb045c.pth)
\| [log](https://download.openmmlab.com/mmrotate/v0.1.0/r3det/r3det_r50_fpn_1x_dota_oc/r3det_r50_fpn_1x_dota_oc_20220126_191226.log.json)
|
| ResNet50 (1024,1024,200) | 70.54 | oc | 1x | 3.65 | 13.6 | - | 2
|                        [r3det_atss_r50_fpn_1x_dota_oc](./r3det_atss_r50_fpn_1x_dota_oc.py)
|                                  [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_atss/r3det_atss_r50_fpn_1x_dota_oc/r3det_atss_r50_fpn_1x_dota_oc-4a3034cd.pth)
\| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_atss/r3det_atss_r50_fpn_1x_dota_oc/r3det_atss_r50_fpn_1x_dota_oc_20220416_171200.log.json)
|

Notes:

- `hbb` means the input of the assigner is the predicted box and the horizontal box that can surround the GT. `obb`
  means the input of the assigner is the predicted box and the GT. They can be switched by `assign_by_circumhbbox`
  in `RotatedRetinaHead`.

## Citation

```
@inproceedings{zhang2020bridging,
  title={Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection},
  author={Zhang, Shifeng and Chi, Cheng and Yao, Yongqiang and Lei, Zhen and Li, Stan Z},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9759--9768},
  year={2020}
}
```
