## AD-Toolbox

Aerial Detection Toolbox based on MMDetection and MMRotate.

Mainly for my own convenience, no need to modify code in mmdet and mmrotate.

English | [简体中文](README_zh-CN.md)

## Supported Datasets

|                        Dataset                        | HBB Det | OBB Det | Instance Segm |
| :---------------------------------------------------: | :-----: | :-----: | :-----------: |
| [DOTA](https://captain-whu.github.io/DOTA/index.html) |    ×    |  ✓ \*   |       -       |
|                    [HRSC2016](<>)                     |    ×    |  ✓ \*   |       -       |
|    [DIOR](https://gcheng-nwpu.github.io/#Datasets)    |    ✓    |    ✓    |       -       |
|   [VHR-10](https://gcheng-nwpu.github.io/#Datasets)   |    ✓    |    ×    |       ✓       |
|   [MAR20](https://gcheng-nwpu.github.io/#Datasets)    |    ✓    |    ✓    |       -       |

**Noting:**

- `*` : `DOTA` and `HRSC2016` are officially support by MMRotate.
- All configs still need test and benchmark.

## Install

Please use the mmrotate in https://github.com/liuyanyi/mmrotate/tree/ryolox
before yolox merged into master.

## Models

- Rotated YOLOX (Will be merged into MMRotate! Pr is in progress https://github.com/open-mmlab/mmrotate/pull/409)
- Models in MMdetection and MMRotate.

## Dependencies

- [MMCV-Full](https://github.com/open-mmlab/mmcv)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMRotate](https://github.com/open-mmlab/mmrotate)
